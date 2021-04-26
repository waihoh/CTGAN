import numpy as np
import torch
from packaging import version
from torch import optim
from torch.nn import functional

from ctgan.conditional import ConditionalGenerator
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from ctgan.sampler import Sampler
from ctgan.transformer import DataTransformer
from torchsummary import summary

from ctgan.config import ctgan_setting as cfg
from ctgan.logger import Logger

### added for validation
from sklearn.model_selection import train_test_split
import ctgan.metric as M
import optuna

########################################Defining our models ################################################################
class Discriminator(Module):
    # Note: The lambda_ is based on WGAN + gradient penalty.
    # See Algorithm 1 in Gulrajani et. al. (2017)
    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        # real_data.size(0) is batch size, eg. 500
        # real_data.size(1) is number of columns, eg. 15
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)  # eg. ([50, 1, 1])
        # duplicates alpha. For each alpha, # cols is real_data.size(1), # rows is pac.
        alpha = alpha.repeat(1, pac, real_data.size(1))  # eg. [(50, 10 , 15)]
        # change shape so that alpha is the same dimension as real_data and fake_data.
        alpha = alpha.view(-1, real_data.size(1))  # eg[(500, 15)]

        # Element-wise multiplication.
        # real_data.shape == fake_data.shape == interpolates.shape == eg. ([500, 15])
        # Note: See section 4 of Gulrajani et. al. (2017), Sampling distribution
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        # Note interpolates passes through the Discriminator forward function.
        disc_interpolates = self(interpolates)  # disc_interpolates.shape == eg. ([50, 1])

        # Computes and returns the sum of gradients of outputs w.r.t. the inputs.
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # gradients.shape == [(500, 15)]
        gradient_penalty = ((
            # reshape to pac * real_data.size(1) sums all
            # the norm is a Frobenius norm.
            # It sums over all interpolates/gradients multiplied to same alpha previously.
            gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_
        return gradient_penalty

    def __init__(self, input_dim, dis_dims, pack=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        print('Dropout rate: ', cfg.DROPOUT)
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(cfg.DROPOUT)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        # NOTE: disable assert so that model_summary can be printed.
        # this is because batch_size of input x is hardcoded in torchsummary.py.
        # See row 60 of torchsummary.py in torchsummary library
        # See also if model_summary in synthesizer.py
        # instead, this is imposed in synthesizer.py instead.
        # See assert self.batch_size % self.pack == 0.
        # assert input.size()[0] % self.pack == 0

        # input.view reshapes the input data by dividing the 1st dim, i.e. batch size
        # and group the data in concatenate manner in 2nd dim
        # example, if input dim is ([500, 15]) and pack is 10,
        # then it is reshaped to ([500/10, 15*10)] = ([50, 150])
        return self.seq(input.view(-1, self.packdim))


class Residual(Module): #####concatenating the input and output together
    # NOTE: a Residual layer will be created for each one of the values in gen_dims provided
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        # concatenate the columns. See 4.4 of Xu et. al (2019),
        # where h2 concat h1 concat h0 before passing through last FCs to generate alpha, beta and d.
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data

#######################Creating the model ##########################
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
################Initialising hyperparameters from config.py ###################################################
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
        self.pack = pack  # Default value of Discriminator pac.
        self.logger = Logger()
        #self.validation_KLD = []
        self.generator_loss = []
        self.discriminator_loss = []
        self.optuna_metric = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.
        ##For one-hot encoding, the authors use this instead of softmax

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

#############Apply activation function #######################################################
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
###########################Calculate conditional loss ##############################
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


#####################Fitting our model ############################################
    def fit(self, data, discrete_columns=tuple(),
            model_summary=False, trans="VGM",
            trial=None, transformer=None, in_val_data=None,
            reload=False):
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

        self.logger.write_to_file('Generator learning rate: ' + str(cfg.GENERATOR_LEARNING_RATE))
        self.logger.write_to_file('Discriminator learning rate: ' + str(cfg.DISCRIMINATOR_LEARNING_RATE))
        self.logger.write_to_file('Batch size: ' + str(cfg.BATCH_SIZE))
        self.logger.write_to_file('Number of Epochs: '+ str(cfg.EPOCHS))
        self.logger.write_to_file('Embedding: ' + str(cfg.EMBEDDING))
        self.logger.write_to_file('Depth: ' + str(cfg.DEPTH))
        self.logger.write_to_file('Width: ' + str(cfg.WIDTH))
        self.logger.write_to_file('Dropout rate: ' + str(cfg.DROPOUT))
        self.logger.write_to_file('Discriminator step: ' + str(cfg.DISCRIMINATOR_STEP))
        self.logger.write_to_file('use cond. gen. ' + str(cfg.CONDGEN))

        self.trans = trans

        if reload:
            self.trained_epoches = 0

        if transformer is None:
            # NOTE: data is split to train:validation:test with 70:15:15 rule
            # Test data has been partitioned outside of this code.
            # The next step is splitting the reamining data to train:validation.
            # Validation data is approximately 17.6%.
            temp_test_size = 15 / (70 + 15)  # 0.176
            exact_val_size = int(temp_test_size * data.shape[0])
            exact_val_size -= exact_val_size % self.pack
            assert exact_val_size % self.pack == 0

            train_data, val_data = train_test_split(data, test_size=exact_val_size, random_state=42)

            if not reload:
                if not hasattr(self, "transformer"):
                    self.transformer = DataTransformer()
                self.transformer.fit(data, discrete_columns, self.trans)
            train_data = self.transformer.transform(train_data)
        else:
            # transformer has been saved separately.
            # input data should have been transformed as well.
            self.transformer = transformer
            train_data = data

            if in_val_data is None:
                ValueError('Validation data must be provided')

            # val_data is not transformed. For computation of KLD.
            val_data = in_val_data

#################Sample real data ############################################################
        data_sampler = Sampler(train_data, self.transformer.output_info, trans=self.trans)

        data_dim = self.transformer.output_dimensions
        self.logger.write_to_file('data dimension: ' + str(data_dim))

        if not reload:
            if not hasattr(self, "cond_generator"):
                self.cond_generator = ConditionalGenerator(
                    train_data,
                    self.transformer.output_info,
                    self.log_frequency,
                    trans=self.trans,
                    use_cond_gen=cfg.CONDGEN
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
        # NOTE: in Discriminator forward function,
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
########################Start training ####################################################
        for i in range(self.epochs):
            self.generator.train() ##switch to train mode
            self.trained_epoches += 1
            for id_ in range(steps_per_epoch):

                for n in range(self.discriminator_steps):
###################### 1. Initialise fake data ###################################################
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self.cond_generator.sample(self.batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = data_sampler.sample(self.batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self.device)
                        m1 = torch.from_numpy(m1).to(self.device)
#################### 2. Add the conditional vector #########################################
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self.batch_size)
                        np.random.shuffle(perm)
    ###################### 3. Sample real data #########################################################
                        real = data_sampler.sample(self.batch_size, col[perm], opt[perm]) #sampling rows and columns according to cond vector
                        c2 = c1[perm] #shuffle the columns

###################### 4. Create synthetic data ###############################################
                    fake = self.generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self.device)

                    if c1 is not None: #cat is referred to as the conditional vector concatenated to the data
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fake
##################### 5. Attribute score from critic############################################
                    y_fake = self.discriminator(fake_cat)
                    y_real = self.discriminator(real_cat)
#################### 6. Calculate loss ##########################################################
                    pen = self.discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self.device)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake)) #Wasserstein's distance between real and fake
#################### 7. Update critic weights and bias using Adam Optimizer ################################
                    self.optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    self.optimizerD.step() #Training for critic stops here

####################8. Create new synthesised data to train the generator ##########################################
                fakez = torch.normal(mean=mean, std=std) # white noise
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez) #Create fake data from white noise
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)
################## 10. Calculate generator loss ######################################################
                loss_g = -torch.mean(y_fake) + cross_entropy #torch.mean(y_real) is zero
################## 11. Update weights and bias for generator ##########################################
                self.optimizerG.zero_grad()
                loss_g.backward()
                self.optimizerG.step()

            self.generator_loss.append(loss_g.detach().cpu())
            self.discriminator_loss.append(loss_d.detach().cpu())
            self.logger.write_to_file("Epoch " + str(self.trained_epoches) +
                                      ", Loss G: " + str(loss_g.detach().cpu().numpy()) +
                                      ", Loss D: " + str(loss_d.detach().cpu().numpy()),
                                      toprint=True)

            # Use Optuna for hyper-parameter tuning (Euclidean KLD)
            if trial is not None:
                # synthetic data by the generator for each epoch
                sampled_train = self.sample(val_data.shape[0], condition_column=None, condition_value=None)
                KL_val_loss = M.KLD(val_data, sampled_train,  discrete_columns)

                # Euclidean distance of KLD
                self.optuna_metric = np.sqrt(np.nansum(KL_val_loss ** 2))
                trial.report(self.optuna_metric, i)

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
########################### 12. Save the model #######################################################################################
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
