import numpy as np
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
# from torch.nn import functional

from torch.optim import Adam
# from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from ctgan.transformer import DataTransformer
from ctgan.conditional import ConditionalGenerator
from ctgan.sampler import Sampler
from ctgan.synthesizer import CTGANSynthesizer  # use _gumbel_softmax

from ctgan.config import tvae_setting as cfg
from ctgan.logger import Logger

### added for validation
from sklearn.model_selection import train_test_split
import ctgan.metric as M
import optuna

class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma


def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    # Evidence loss lower bound
    # See equation 10 in Kingma and Welling 2013.
    # See also useful information in https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf
    st = 0
    loss = []

    # loss of decoder
    for item in output_info:
        # negative log Gaussian likelihood lost for alpha
        if item[1] == 'tanh':
            ed = st + item[0]
            std = sigmas[st]
            loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2)).sum())
            loss.append(torch.log(std) * x.size()[0])
            st = ed
        # cross entropy loss for
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(cross_entropy(
                recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
            st = ed
        else:
            assert 0



    assert st == recon_x.size()[1]

    # loss of encoder.
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return average loss per batch
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAESynthesizer(object):
    """TVAESynthesizer."""

    def __init__(self, l2scale=1e-5, trained_epoches = 0, log_frequency=True):

        self.embedding_dim = cfg.EMBEDDING
        self.compress_dims = np.repeat(cfg.WIDTH, cfg.DEPTH)
        self.decompress_dims = np.repeat(cfg.WIDTH, cfg.DEPTH)
        self.log_frequency = log_frequency
        self.l2scale = l2scale
        self.batch_size = cfg.BATCH_SIZE
        self.epochs = cfg.EPOCHS
        self.lr = cfg.LEARNING_RATE
        self.loss_factor = 1  # 2 TODO: why 2 in original code? Should be 1 based on loss function.
        self.trained_epoches = trained_epoches

        # exponential moving average of latent space, mu and sigma
        # use these values to sample from N(ema_mu, ema_sig**2) iso N(0,1)
        self.ema_fraction = 0.9
        self.ema_mu = 0
        self.ema_std = 0

        self.logger = Logger()
        self.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"

        self.use_cond_gen = cfg.CONDGEN
        self.validation_KLD = []
        self.total_loss = []
        self.threshold = None
        self.prop_dis_validation = None
        self.trial_completed = True

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
                transformed = CTGANSynthesizer()._gumbel_softmax(data[:, st:ed], tau=0.2)
                data_t.append(transformed)
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def fit(self, data, discrete_columns=tuple(),
            model_summary=False, trans="VGM",
            trial=None, transformer=None, in_val_data=None, threshold=None):

        self.logger.write_to_file('Learning rate: ' + str(self.lr))
        self.logger.write_to_file('Batch size: ' + str(self.batch_size))
        self.logger.write_to_file('Number of Epochs: ' + str(self.epochs))
        self.logger.write_to_file('Use conditional vector: ' + str(self.use_cond_gen))

        self.trans = trans

        if transformer is None:
            # data is split to train:validation:test with 70:15:15 rule
            # test data has been partitioned outside of this code.
            # thus, we split data to train:validation. Validation data is approximately 17.6%.
            temp_test_size = 15 / (70 + 15)  # 0.176
            exact_val_size = int(temp_test_size * data.shape[0])

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
                use_cond_gen=self.use_cond_gen
            )

        # NOTE: these steps are different from ctgan
        # dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Note: vectors from conditional generator are appended latent space
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim+self.cond_generator.n_opt, self.compress_dims, data_dim).to(self.device)

        if model_summary:
            print("*" * 100)
            print("ENCODER")
            summary(self.encoder, (data_dim, ))
            print("*" * 100)

            print("DECODER")
            summary(self.decoder, (self.embedding_dim+self.cond_generator.n_opt, ))
            print("*" * 100)

        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr,
            weight_decay=self.l2scale)
        #print(optimizerAE)
        assert self.batch_size % 2 == 0

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)

        for i in range(self.epochs):
            self.decoder.train() ##switch to train mode
            self.trained_epoches += 1
            for id_ in range(steps_per_epoch):
                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                optimizerAE.zero_grad()
                real = torch.from_numpy(real.astype('float32')).to(self.device)

                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu

                # NEW. 2020-12-03.
                # compute exponetial moving average
                self.ema_mu = self.ema_fraction * mu.mean() + (1 - self.ema_fraction) * self.ema_mu
                self.ema_std = self.ema_fraction * std.mean() + (1 - self.ema_fraction) * self.ema_std

                # NEW
                # Conditional vector is added to latent space.
                if c1 is not None:
                    emb = torch.cat([emb, c2], dim=1)
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = loss_function(
                    rec, real, sigmas, mu, logvar, self.transformer.output_info, self.loss_factor)
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
            self.total_loss.append(loss.detach().cpu())
            self.logger.write_to_file("Epoch " + str(self.trained_epoches) +
                                      ", Loss: " + str(loss.detach().cpu().numpy()),
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
                    self.trial_completed = False
                    raise optuna.exceptions.TrialPruned()

    def sample(self, samples, condition_column=None, condition_value=None):
        self.decoder.eval()

        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.covert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self.cond_generator.generate_cond_from_condition_column_info(
                condition_info, self.batch_size)
        else:
            global_condition_vec = None

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            # print("ema_mu, ema_std", self.ema_mu, self.ema_std)
            # NOTE: Instead of using N(0,1), we use the mean and std 'learnt' during encoding.
            # i.e. N(self.ema_mu, self.ema_std**2).
            # It is nonetheless observed from tests that ema_mu and ema_std are close to 0 and 1 respectively.
            # mean = torch.zeros(self.batch_size, self.embedding_dim)
            # std = mean + 1

            # Added to(self.device) to mean and std
            # so that they can be added with self.ema_mu and self.ema_std respectively
            mean = torch.zeros(self.batch_size, self.embedding_dim).to(self.device) + self.ema_mu
            std = torch.zeros(self.batch_size, self.embedding_dim).to(self.device) + self.ema_std
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

            fake, sigmas = self.decoder(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def save(self, path):
        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        model.encoder.to(model.device)
        model.decoder.to(model.device)

        return model
