import numpy as np
import torch
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, LeakyReLU, Module, ReLU, Sequential, Sigmoid, init
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
# from torch.utils.data import DataLoader, TensorDataset
from ctgan.transformer import DataTransformer
from torchsummary import summary
from ctgan.conditional import ConditionalGenerator
from ctgan.sampler import Sampler
from ctgan.synthesizer import CTGANSynthesizer  # use _gumbel_softmax

from ctgan.config import tablegan_setting as cfg
# NOTE: Added conditional generator to the code.

class Discriminator(Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input_):
        return self.seq(input_)


## used for classification problem
## may not be used in OVS dataset
class Classifier(Module):
    def __init__(self, meta, side, layers, device):
        super(Classifier, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        self.valid = True
        # if meta[-1]['name'] != 'label' or meta[-1]['type'] != CATEGORICAL or meta[-1]['size'] != 2:
        if meta[-1]['name'] != 'label':  ##check whether the last column is "label"
            self.valid = False

        masking = np.ones((1, 1, side, side), dtype='float32')
        index = len(self.meta) - 1
        self.r = index // side
        self.c = index % side
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking).to(device)

    def forward(self, input):
        label = (input[:, :, self.r, self.c].view(-1) + 1) / 2
        input = input * self.masking.expand(input.size())
        return self.seq(input).view(-1), label


def determine_layers(side, random_dim, num_channels, dlayer):
    """
    Args:
        side: length of square matrix
        random_dim: dim of z vector
        num_channels: number of filters / feature maps
        dlayer: 0: no changes. -1: remove last item in layer_dims, 1: add a 1X1 Convolution layer.

    Returns:
        lists of layers in Discriminator, Generator and Classifier.
    """

    assert side >= 4 and side <= 64  ##change to 64 for OVS dataset

    scale_factor = cfg.SCALE_FACTOR  # ini value: 2
    kernel_size = cfg.KERNEL_SIZE # 4  # ini value: 4
    stride = cfg.STRIDE # 2  # ini value: 2

    # layer_dims = [(1, side), (num_channels, side // 2)]
    layer_dims = [(1, side), (num_channels, side // scale_factor)]

    #while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
    while layer_dims[-1][1] > 3 and len(layer_dims) < 5: ## max of 5 for the case side = 64
        # layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
        layer_dims.append((layer_dims[-1][0] * scale_factor, layer_dims[-1][1] // scale_factor))

    # WH: Remove last layer
    if dlayer == -1:
        layer_dims.pop()

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
           ## Conv2d(in_channels = prev[0], out_channels = curr[0], kernel_size = 4,
           ## stride = 2, padding = 1, bias=False)
            Conv2d(prev[0], curr[0], kernel_size, stride, 1, bias=False),
            BatchNorm2d(curr[0]),
            ## the slope of the leak was set to 0.2
            LeakyReLU(0.2, inplace=True) ##y=0.2x when x<0
        ]

    # WH: Add a 1X1 convolution layer to increase depth of network
    if dlayer == 1:
        layers_D += [Conv2d(layer_dims[-1][0], layer_dims[-1][0], 1, 1, 0),
                     BatchNorm2d(layer_dims[-1][0]),
                     LeakyReLU(0.2, inplace=True)
                     ]

    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Sigmoid()
    ]

    layers_G = [
        # ConvTranspose2d(
        #     in_channels = random_dim, out_channels = layer_dims[-1][0], kernel_size = layer_dims[-1][1],
        #     stride = 1, padding = 0, output_padding=0, bias=False)
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    # WH: Add a 1X1 convolution layer to increase depth of network
    if dlayer == 1:
        layers_G += [BatchNorm2d(layer_dims[-1][0]),
                     ReLU(True),
                     ConvTranspose2d(layer_dims[-1][0], layer_dims[-1][0], 1, 1, 0, output_padding=0, bias=False)
                     ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
        ## ConvTranspose2d(in_channels = prev[0], out_channels = curr[0], kernel_size = 4,
        ## stride = 2, padding = 1, output_padding=0, bias=True)
            ConvTranspose2d(prev[0], curr[0], kernel_size, stride, 1, output_padding=0, bias=True)
        ]
    #layers_G += [Tanh()] ##revmoved and use _apply_activate function instead

    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [
            Conv2d(prev[0], curr[0], kernel_size, stride, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]

    layers_C += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0)]

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        ###All weights for convolutional and de-convolutional layers were initialized
        # from a zero-centered Normal distribution with standard deviation 0.02.
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


def get_side(total_dims):
    output = 0
    sides = [4, 8, 16, 24, 32, 64]  # added 64 to accommodate OVS dataset
    for i in sides:
        if i * i >= total_dims:
            output = i
            break
    return output


def reshape_data(data, side):
    data = data.copy().astype('float32')
    if side * side > len(data[1]):
        padding = np.zeros((len(data), side * side - len(data[1])))
        data = np.concatenate([data, padding], axis=1)
    return data.reshape(-1, 1, side, side)


class TableganSynthesizer(object):
    """docstring for TableganSynthesizer??"""

    # def __init__(self,
    #              random_dim=100,
    #              num_channels=64,
    #              l2scale=1e-5,
    #              batch_size=500):
    def __init__(self,
                     random_dim=cfg.EMBEDDING,
                     num_channels=cfg.NUM_CHANNELS,
                     l2scale=1e-5,
                     batch_size=cfg.BATCH_SIZE,
                     dlayer = cfg.DLAYER
                ):

        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self.dlayer = dlayer

        self.batch_size = batch_size
        self.trained_epoches = 0
        self.side = 0
        self.data_dim = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_activate(self, data, padding = True):
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
        ### TODO: How to deal with those values padded with 0
        ### cannot replaced by 0; try to apply gumbel_softmax first
        if padding:
            transformed0 = CTGANSynthesizer()._gumbel_softmax(data[:, self.transformer.output_dimensions:data.shape[1]], tau=0.2)
            data_t.append(transformed0)
        return torch.cat(data_t, dim=1)

    def get_noise_real(self, get_actual_data=False):
        noise = torch.randn(self.batch_size, self.random_dim, device=self.device)
        real = None

        condvec = self.cond_generator.sample(self.batch_size)
        if condvec is None:
            c1, m1, col, opt = None, None, None, None
            if get_actual_data:
                real = self.data_sampler.sample(self.batch_size, col, opt)
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self.device)

            perm = np.arange(self.batch_size)
            np.random.shuffle(perm)
            if get_actual_data:
                real = self.data_sampler.sample(self.batch_size, col[perm], opt[perm])
            c2 = c1[perm]
            noise = torch.cat([noise, c2], dim=1)

        # Add 2 dimensions at the back: final dims: batch_size x (random_dim + cond_generator.n_opt) x 1 x 1
        noise = noise.unsqueeze(-1)
        noise = noise.unsqueeze(-1)

        return noise, real

    # def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple(), epochs=300):
    def fit(self, data, discrete_columns=tuple(), epochs=cfg.EPOCHS, log_frequency=True,
            model_summary=False, trans="VGM", use_cond_gen=True):
        print('Learning rate: ', cfg.LEARNING_RATE)
        print('Batch size: ', self.batch_size)
        print('Number of Epochs: ', cfg.EPOCHS)
        print('Depth of layer: ',self.dlayer)
        self.trans = trans
        # sides = [4, 8, 16, 24, 32]
        # for i in sides:
        #     if i * i >= data.shape[1]:
        #         self.side = i
        #         break

        # self.transformer = TableganTransformer(self.side)
        # self.transformer.fit(data, categorical_columns, ordinal_columns)
        # data = self.transformer.transform(data)

        # NOTE:
        # we'll use transformer.transform function. The output data is 1D instead of 2D.
        # we'll reshape the data later.
        if not hasattr(self, "transformer"):
            self.transformer = DataTransformer()
            self.transformer.fit(data, discrete_columns, self.trans)
        data = self.transformer.transform(data)

        print('data shape', data.shape)

        self.data_sampler = Sampler(data, self.transformer.output_info, trans=self.trans)

        # NOTE: changed data_dim to self.data_dim. It'll be used later in sample function.
        self.data_dim = self.transformer.output_dimensions
        print('data dim', self.data_dim)

        if not hasattr(self, "cond_generator"):
            self.cond_generator = ConditionalGenerator(
                data,
                self.transformer.output_info,
                log_frequency,
                trans=self.trans,
                use_cond_gen=use_cond_gen
            )

        # compute side after transformation
        self.side = get_side(self.data_dim)
        print('side', self.side)

        # data = torch.from_numpy(data.astype('float32')).to(self.device)
        # dataset = TensorDataset(data)
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(
            self.side, self.random_dim + self.cond_generator.n_opt, self.num_channels, self.dlayer)

        self.generator = Generator(self.transformer.meta, self.side, layers_G).to(self.device)
        self.discriminator = Discriminator(self.transformer.meta, self.side, layers_D).to(self.device)
        self.classifier = Classifier(
            self.transformer.meta, self.side, layers_C, self.device).to(self.device)

        if model_summary:
            print("*" * 100)
            print("GENERATOR")
            # in determine_layers, see side//2.
            summary(self.generator,
                    (self.random_dim + self.cond_generator.n_opt, self.side // 2, self.side // 2))
            print("*" * 100)

            print("DISCRIMINATOR")
            summary(self.discriminator, (1, self.side, self.side))
            print("*" * 100)

            # print("CLASSIFIER")
            # summary(self.classifier, (1, self.side, self.side))
            # print("*" * 100)

        ##learning rate is 0.0002
        optimizer_params = dict(lr=cfg.LEARNING_RATE, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params) ##nn.parameters() returns the trainable parameters
        optimizerD = Adam(self.discriminator.parameters(), **optimizer_params)
        optimizerC = Adam(self.classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.classifier.apply(weights_init)

        steps_per_epoch = max(len(data) // self.batch_size, 1)
        for i in range(epochs):
            self.trained_epoches += 1
            for id_ in range(steps_per_epoch):
                noise, real = self.get_noise_real(True) ## cond is added
                fake = self.generator(noise)
                ## reshape to vector then apply activate function
                fake = torch.reshape(fake,(self.batch_size,self.side * self.side))
                fake = self._apply_activate(fake,True)
                ## reshape to 2D.
                fake = torch.reshape(fake, (self.batch_size, 1, self.side, self.side))
                # Use reshape function to add zero padding and reshape to 2D.
                real = reshape_data(real, self.side)
                real = torch.from_numpy(real.astype('float32')).to(self.device)

                optimizerD.zero_grad()
                y_real = self.discriminator(real)
                y_fake = self.discriminator(fake)
                ## L_orig^D
                loss_d = (
                    -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                loss_d.backward()
                optimizerD.step()

                #  To train the generator with L_orig^G first
                noise, _ = self.get_noise_real(False)
                # noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                fake = torch.reshape(fake, (self.batch_size, self.side * self.side))
                fake = self._apply_activate(fake,True)
                fake = torch.reshape(fake, (self.batch_size, 1, self.side, self.side))

                optimizerG.zero_grad()
                y_fake = self.discriminator(fake)
                ## L_orig^G
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True) ##by setting retain_graph = True, generator is trained by L_orig^G+L_info^G
                ## L_mean in eq(2)
                loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0), 1)
                ## L_sd in eq (3)
                loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0), 1)
                ## L_info in eq (4) with delta_mean = 0 and delta_sd =0
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                # noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                # fake = self.generator(noise)
                if self.classifier.valid:
                    noise, real = self.get_noise_real(True)
                    fake = self.generator(noise)
                    fake = torch.reshape(fake, (self.batch_size, self.side * self.side))
                    fake = self._apply_activate(fake,True)
                    fake = torch.reshape(fake, (self.batch_size, 1, self.side, self.side))

                    real_pre, real_label = self.classifier(real)
                    fake_pre, fake_label = self.classifier(fake)

                    loss_cc = binary_cross_entropy_with_logits(real_pre, real_label)
                    loss_cg = binary_cross_entropy_with_logits(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()
                    loss_c = (loss_cc, loss_cg)
                else:
                    loss_c = None

                # if (id_ + 1) % 50 == 0:
                if (id_ + 1) % 1 == 0:
                    print("epoch", i + 1, "step", id_ + 1, loss_d, loss_g, loss_c, flush=True)


  ### following ctgan and tvae, added the parts updated by the authors.
    def sample(self, n, condition_column=None, condition_value=None):
    #def sample(self, n):
        self.generator.eval()

        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.covert_column_name_value_to_id(condition_column, condition_value)
            global_condition_vec = self.cond_generator.generate_cond_from_condition_column_info(condition_info, self.batch_size)
        else:
            global_condition_vec = None

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            # noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
            #noise, _ = self.get_noise_real(False)
            noise = torch.randn(self.batch_size, self.random_dim, device=self.device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self.cond_generator.sample_zero(self.batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                noise = torch.cat([noise, c1], dim=1)

            # Add 2 dimensions at the back: final dims: batch_size x (random_dim + cond_generator.n_opt) x 1 x 1
            noise = noise.unsqueeze(-1)
            noise = noise.unsqueeze(-1)

            fake = self.generator(noise)
            ## reshape to vector then apply activate function
            fake = torch.reshape(fake, (self.batch_size, self.side * self.side))
            fake = self._apply_activate(fake,False)
            ## no need to reshape to 2D here
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        # return self.transformer.inverse_transform(data[:n])

        # # 2020-11-12
        # # first, reshape the square matrices to 1D
        #data = data[:n].reshape(-1, self.side * self.side)
        print('inverse_transform_tablegan, after reshaping to 1D:', data.shape)
        # # second, remove the padded values.
        # # however, this line does not seem to be required.
        # # it'll work just fine by calling inverse_transform directly.
        # data = data[:, :self.data_dim]
        # print('inverse_transform_tablegan, after slicing:', data.shape)

       # return self.transformer.inverse_transform(data[:n], None)
        return self.transformer.inverse_transform(data, None)

    def save(self, path):
        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.classifier.to(self.device)

        torch.save(self, path) ##saving the entire model
       ## torch.save(self.state_dict(),path) ##save only parameters
        ### load this model
        ## model = torch.load(path)
        ## print(model)

        self.device = device_bak
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.classifier.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.generator.to(model.device)
        model.discriminator.to(model.device)
        model.classifier.to(model.device)

        return model
