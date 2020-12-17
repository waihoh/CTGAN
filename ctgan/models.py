import torch
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from ctgan.config import ctgan_setting as cfg


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
            #seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
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


class Residual(Module):
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
