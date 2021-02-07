import numpy as np
import torch
from packaging import version
from torch import optim
from torch.nn import functional

from ctgan.conditional import ConditionalGenerator
from ctgan.models import Discriminator, Generator
from ctgan.sampler import Sampler
from ctgan.transformer import DataTransformer
from torchsummary import summary

from ctgan.config import ctgan_setting as cfg
from ctgan.logger import Logger

### added for validation
from sklearn.model_selection import train_test_split
import ctgan.metric as M
import optuna


class CTGANSynthesizer(object):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        gen_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        dis_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        l2scale (float):
            Weight Decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
    """


    def __init__(self, l2scale=1e-6, pack = 10, log_frequency=True):
        self.embedding_dim = cfg.EMBEDDING
        self.gen_dim = np.repeat(cfg.WIDTH, cfg.DEPTH)
        self.dis_dim = np.repeat(cfg.WIDTH, cfg.DEPTH)

        self.l2scale = l2scale
        self.batch_size = cfg.BATCH_SIZE
        self.epochs = cfg.EPOCHS
        self.glr = cfg.GENERATOR_LEARNING_RATE
        self.dlr = cfg.DISCRIMINATOR_LEARNING_RATE
        self.log_frequency = log_frequency
        self.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        self.trained_epoches = 0
        self.discriminator_steps = cfg.DISCRIMINATOR_STEP
        self.pack = pack  # Default value of Discriminator pac. See models.py
        self.logger = Logger()
        self.validation_KLD = []
        self.generator_loss = []
        self.discriminator_loss = []
        self.threshold = None
        self.prop_dis_validation = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """

        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                data_t.append(transformed)
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        skip = False
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                st += item[0]
                if self.trans == "VGM":
                    skip = True

            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                ed_c = st_c + item[0]
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction='none'
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

            else:
                assert 0

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    #def fit(self, threshold, data,  discrete_columns=tuple(), model_summary=False, trans="VGM", use_cond_gen=True,trial=None):
    def fit(self, data, discrete_columns=tuple(),
            model_summary=False, trans="VGM", use_cond_gen=True,
            trial=None, transformer=None, in_val_data=None, threshold=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            data (numpy.ndarray or pandas.DataFrame):
                Whole Data. It must be a 2-dimensional numpy array or a
                pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of training epochs. Defaults to 300.
        """

        self.logger.write_to_file('Generator learning rate: ' + str(self.glr))
        self.logger.write_to_file('Discriminator learning rate: ' + str(self.dlr))
        self.logger.write_to_file('Batch size: ' + str(self.batch_size))
        self.logger.write_to_file('Number of Epochs: '+ str(self.epochs))

        self.trans = trans

        if transformer is None:
            # data is split to train:validation:test with 70:15:15 rule
            # test data has been partitioned outside of this code.
            # thus, we split data to train:validation. Validation data is approximately 17.6%.
            temp_test_size = 15 / (70 + 15)  # 0.176
            exact_val_size = int(temp_test_size * data.shape[0])
            exact_val_size -= exact_val_size % self.pack
            assert exact_val_size % self.pack == 0

            train_data, val_data = train_test_split(data, test_size=exact_val_size, random_state=42)

            if not hasattr(self, "transformer"):
               self.transformer = DataTransformer()
               self.transformer.fit(data, discrete_columns, self.trans)
               train_data = self.transformer.transform(train_data)
        else:
            # transformer has been saved separately.
            # input data should have been transformed as well.
            self.transformer = transformer
            train_data = data
            val_data = in_val_data

        data_sampler = Sampler(train_data, self.transformer.output_info, trans=self.trans)

        data_dim = self.transformer.output_dimensions
        self.logger.write_to_file('data dimension: ' + str(data_dim))

        if not hasattr(self, "cond_generator"):
            self.cond_generator = ConditionalGenerator(
                train_data,
                self.transformer.output_info,
                self.log_frequency,
                trans=self.trans,
                use_cond_gen=use_cond_gen
            )

        if not hasattr(self, "generator"):
            self.generator = Generator(
                self.embedding_dim + self.cond_generator.n_opt,
                self.gen_dim,
                data_dim
            ).to(self.device)

        if not hasattr(self, "discriminator"):
            self.discriminator = Discriminator(
                data_dim + self.cond_generator.n_opt,
                self.dis_dim,
                pack=self.pack
            ).to(self.device)

        if not hasattr(self, "optimizerG"):
            self.optimizerG = optim.Adam(
                # self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9),
                self.generator.parameters(), lr=self.glr, betas=(0.5, 0.9),
                weight_decay=self.l2scale
            )

        if not hasattr(self, "optimizerD"):
            self.optimizerD = optim.Adam(
                # self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
                self.discriminator.parameters(), lr=self.dlr, betas=(0.5, 0.9))

        # assert self.batch_size % 2 == 0
        # NOTE: in models.py, Discriminator forward function,
        # there is a reshape of input data, i.e. input.view() that is dependent on self.pack.
        assert self.batch_size % self.pack == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        if model_summary:
            print("*" * 100)
            print("GENERATOR")
            summary(self.generator, (self.embedding_dim + self.cond_generator.n_opt,))
            print("*" * 100)

            print("DISCRIMINATOR")
            this_size = (data_dim + self.cond_generator.n_opt)*self.pack
            summary(self.discriminator, (this_size,))
            print("*" * 100)

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)

        for i in range(self.epochs):
            self.generator.train() ##switch to train mode
            self.trained_epoches += 1
            for id_ in range(steps_per_epoch):

                for n in range(self.discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self.cond_generator.sample(self.batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = data_sampler.sample(self.batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self.device)
                        m1 = torch.from_numpy(m1).to(self.device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self.batch_size)
                        np.random.shuffle(perm)
                        real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self.generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self.device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fake

                    y_fake = self.discriminator(fake_cat)
                    y_real = self.discriminator(real_cat)

                    pen = self.discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self.device)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    self.optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    self.optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                self.optimizerG.zero_grad()
                loss_g.backward()
                self.optimizerG.step()

            self.generator_loss.append(loss_g.detach().cpu())
            self.discriminator_loss.append(loss_d.detach().cpu())
            self.logger.write_to_file("Epoch " + str(self.trained_epoches) +
                                      ", Loss G: " + str(loss_g.detach().cpu().numpy()) +
                                      ", Loss D: " +str(loss_d.detach().cpu().numpy()),
                                      toprint=False)

            # Use Optuna for hyper-parameter tuning
            # Use KL divergence proportion of dissimilarity as metric (to minimize).
            if trial is not None:
                if self.threshold is None:
                    if threshold is None:
                        self.threshold = M.determine_threshold(data, val_data.shape[0], discrete_columns, n_rep=10)
                    else:
                        self.threshold = threshold

                # synthetic data by the generator for each epoch
                sampled_train = self.sample(val_data.shape[0], condition_column=None, condition_value=None)
                KL_val_loss = M.KLD(val_data, sampled_train,  discrete_columns)
                diff_val = KL_val_loss - self.threshold
                self.validation_KLD.append(KL_val_loss)
                self.prop_dis_validation = np.count_nonzero(diff_val >= 0)/np.count_nonzero(~np.isnan(diff_val))
                trial.report(self.prop_dis_validation, i)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()


    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.generator.eval() ## switch to evaluate mode
        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.covert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self.cond_generator.generate_cond_from_condition_column_info(
                condition_info, self.batch_size)
        else:
            global_condition_vec = None

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self.cond_generator.sample_zero(self.batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, None)

    def save(self, path):
        assert hasattr(self, "generator")
        assert hasattr(self, "discriminator")
        assert hasattr(self, "transformer")

        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        model.generator.to(model.device)
        model.discriminator.to(model.device)
        return model
