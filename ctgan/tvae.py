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
from ctgan.ctgan import CTGANSynthesizer  # use _gumbel_softmax

from ctgan.config import tvae_setting as cfg
from ctgan.logger import Logger

### added for validation
from sklearn.model_selection import train_test_split
import ctgan.metric as M
import optuna

################################### Defining the Neural Networks #########################################

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
        # NOTE: See section 4.5 in Xu et al (2019).
        # In the original code, sigma is assumed and fixed at 0.1
        # It is used in the calculation of loss.
        return self.seq(input), self.sigma


def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor, cond_gen_encoder):
    # Evidence loss lower bound
    # See equation 10 in Kingma and Welling 2013. We need to maximize ELBO in this equation.
    # That's equivalent to minimizing -ELBO, which is coded here.
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
        # cross entropy loss for beta and d
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(cross_entropy(
                recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
            st = ed
        else:
            assert 0

    # conditional vector is used as an input to encoder.
    # inputting the classification error of conditional vector and synthesised data
    if cond_gen_encoder:
        ed = recon_x.size()[1]
        loss.append(cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
        st = ed

    assert st == recon_x.size()[1]

    # loss of encoder.
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return average loss per batch
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

#################################Initialising the TVAE model ##################################################
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
        self.loss_factor = 1  # NOTE: It is 1 based in Kingma and Welling's paper. In CTGAN paper, the authors set it as 2. This can be a hyper-parameter to tune.
        self.trained_epoches = trained_epoches

        # exponential moving average of latent space, mu and sigma
        # use these values to sample from N(ema_mu, ema_sig**2) iso N(0,1)
        self.ema_fraction = 0.9
        self.ema_mu = 0
        self.ema_std = 0

        self.logger = Logger()
        self.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"

        self.cond_gen_encoder = cfg.CONDGEN_ENCODER
        self.cond_gen_latent = cfg.CONDGEN_LATENT
        #self.validation_KLD = []
        self.total_loss = []
        self.val_loss = []
        self.optuna_metric = None

######################## Function to add into the last layer ######################################
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

############################# 1. Fitting the model #######################################################
    def fit(self, data, discrete_columns=tuple(),
            model_summary=False, trans="VGM",
            trial=None, transformer=None, in_val_data=None,
            reload=False):

        self.logger.write_to_file('Learning rate: ' + str(cfg.LEARNING_RATE))
        self.logger.write_to_file('Embedding: ' + str(cfg.EMBEDDING))
        self.logger.write_to_file('Depth: ' + str(cfg.DEPTH))
        self.logger.write_to_file('Width: ' + str(cfg.WIDTH))
        self.logger.write_to_file('Batch size: ' + str(cfg.BATCH_SIZE))
        self.logger.write_to_file('Number of Epochs: ' + str(cfg.EPOCHS))
        self.logger.write_to_file('Loss factor: ' + str(self.loss_factor))
        self.logger.write_to_file('Log frequency: ' + str(self.log_frequency))
        self.logger.write_to_file('Encoder cond. vector: ' + str(self.cond_gen_encoder))
        self.logger.write_to_file('L2 scale: ' + str(self.l2scale))
        self.logger.write_to_file('Latent cond. vector: ' + str(self.cond_gen_latent))

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

            train_data, val_data = train_test_split(data, test_size=exact_val_size, random_state=42)

            if not reload:
                if not hasattr(self, "transformer"):
                    self.transformer = DataTransformer()
                self.transformer.fit(data, discrete_columns, self.trans)
            train_data = self.transformer.transform(train_data)
            val_data_transformed = self.transformer.transform(val_data)
        else:
            # transformer has been saved separately.
            # input data should have been transformed as well.
            self.transformer = transformer
            train_data = data  # load transformed train data

            # When transformer is provided, enforce that validation data is provided.
            # This is important, especially in tvae, since we are computing the validation ELBO loss
            if in_val_data is None:
                ValueError('Validation data must be provided')

            # val_data is not transformed. For computation of KLD.
            val_data = in_val_data
            # next, we transform need a transformed val data for computation of ELBO validation
            val_data_transformed = self.transformer.transform(val_data)

        #Sample the transformed data
        data_sampler = Sampler(train_data, self.transformer.output_info, trans=self.trans)

        data_dim = self.transformer.output_dimensions
        self.logger.write_to_file('data dimension: ' + str(data_dim))

        if not hasattr(self, "cond_generator"):
            self.cond_generator = ConditionalGenerator(
                train_data,
                self.transformer.output_info,
                self.log_frequency,
                trans=self.trans,
                use_cond_gen=(self.cond_gen_encoder or self.cond_gen_latent)
            )

        # NOTE: these steps are different from ctgan
        # dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Note: vectors from conditional generator are appended latent space
        if not reload:
            self.encoder = Encoder(data_dim + self.cond_gen_encoder * self.cond_generator.n_opt,
                                   self.compress_dims,
                                   self.embedding_dim).to(self.device)

            self.decoder = Decoder(self.embedding_dim + self.cond_gen_latent * self.cond_generator.n_opt,
                                   self.compress_dims,
                                   data_dim + self.cond_gen_encoder * self.cond_generator.n_opt).to(self.device)

        if model_summary:
            print("*" * 100)
            print("ENCODER")
            summary(self.encoder, (data_dim + self.cond_gen_encoder * self.cond_generator.n_opt,))
            print("*" * 100)
            print("DECODER")
            summary(self.decoder, (self.embedding_dim + self.cond_gen_latent * self.cond_generator.n_opt,))
            print("*" * 100)

        #Initialise the optimizer
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr,
            weight_decay=self.l2scale)

        assert self.batch_size % 2 == 0

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)

        #Start training
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
###################### 2. Conditional vector is added before encoder. #######################
                if self.cond_gen_encoder:
                    real = torch.cat([real, c2], dim=1)
################### 3. Computing the mu, sigma, logvar ################################
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu

                # NEW. 2020-12-03.
                # compute exponetial moving average
                self.ema_mu = self.ema_fraction * mu.mean() + (1 - self.ema_fraction) * self.ema_mu
                self.ema_std = self.ema_fraction * std.mean() + (1 - self.ema_fraction) * self.ema_std

################### 4. Conditional vector is added into latent space, after the encoder. ###################################
                if self.cond_gen_latent:
                    if c1 is not None:
                        emb = torch.cat([emb, c2], dim=1)
#################### 5. Decoding the latent space to create synthesised data ########################################################
                rec, sigmas = self.decoder(emb)
################### 6. Calculate training loss #################################################################
                loss_1, loss_2 = loss_function(
                    rec, real, sigmas, mu, logvar, self.transformer.output_info, self.loss_factor, self.cond_gen_encoder)
                loss = loss_1 + loss_2
                loss.backward()
################## 7. Updating the weights and biases using Adam Optimizer #######################################################
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
################### 8. Calculate validation loss for TVAE #############################################################
            with torch.no_grad():
                self.encoder.eval()
                self.decoder.eval()

                if self.cond_generator.n_opt > 0:
                    c1_val = torch.zeros(size=(val_data_transformed.shape[0], self.cond_generator.n_opt)).to(
                        self.device)

                this_val_data = torch.from_numpy(val_data_transformed.astype('float32')).to(self.device)
                if self.cond_gen_encoder:
                    this_val_data = torch.cat([this_val_data, c1_val], dim=1)

                mu_val, std_val, logvar_val = self.encoder(this_val_data)
                eps_val = torch.randn_like(std_val)
                emb_val = eps_val * std_val + mu_val


                if self.cond_gen_latent:
                    emb_val = torch.cat([emb_val, c1_val], dim=1)
                rec_val, sigmas_val = self.decoder(emb_val)
                loss_1_val, loss_2_val = loss_function(
                    rec_val, this_val_data, sigmas_val, mu_val, logvar_val,
                    self.transformer.output_info,
                    self.loss_factor, self.cond_gen_encoder)
                val_loss = loss_1_val + loss_2_val
                self.val_loss.append(val_loss.detach().cpu())

                self.encoder.train()
                self.decoder.train()

            self.total_loss.append(loss.detach().cpu().numpy())
            self.logger.write_to_file("Epoch " + str(self.trained_epoches) +
                                      ", Training Loss: " + str(loss.detach().cpu().numpy()) +
                                      ", Validation loss: " + str(val_loss.detach().cpu().numpy()),
                                      toprint=True)

            # Use Optuna for hyper-parameter tuning (Euclidean KLD)
            if trial is not None:
                # synthetic data by the generator for each epoch
                sampled_train = self.sample(val_data.shape[0], condition_column=None, condition_value=None)
                # Euclidean KLD
                KL_val_loss = M.KLD(val_data, sampled_train,  discrete_columns)
                self.optuna_metric = np.sqrt(np.nansum(KL_val_loss ** 2))

                trial.report(self.optuna_metric, i)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
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
            # NOTE: empirically, ema_mu and ema_std are close to 0 and 1 respectively.
            # justifying the original assumptions
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
                if self.cond_gen_latent:
                    fakez = torch.cat([fakez, c1], dim=1)

            fake, sigmas = self.decoder(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())


#########################################Save the model into pkl file (the whole model + training loss + validation loss) #####################
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
