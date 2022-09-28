# import os

# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms

# import pyro
# import pyro.distributions as dist
# import pyro.contrib.examples.util  # patches torchvision
# from pyro.infer import SVI, Trace_ELBO
# from pyro.optim import Adam

# assert pyro.__version__.startswith('1.8.2')
# pyro.distributions.enable_validation(False)

# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader


# class Decoder(nn.Module):
#     def __init__(self, z_dim, hidden_dim, output_dim):
#         super().__init__()
#         # setup the two linear transformations used
#         self.fc1 = nn.Linear(z_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, output_dim)
#         # setup the non-linearities
#         self.softplus = nn.Softplus()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, z):
#         # define the forward computation on the latent z
#         # first compute the hidden units
#         hidden = self.softplus(self.fc1(z))
#         # return the parameter for the output Bernoulli
#         # each is of size batch_size x 784
#         loc_img = self.sigmoid(self.fc21(hidden))
#         return loc_img


# class Encoder(nn.Module):
#     def __init__(self, z_dim, hidden_dim, input_dim):
#         super().__init__()
#         # setup the three linear transformations used
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, z_dim)
#         self.fc22 = nn.Linear(hidden_dim, z_dim)
#         # setup the non-linearities
#         self.softplus = nn.Softplus()
#         self.input_dim = input_dim

#     def forward(self, x):
#         # define the forward computation on the image x
#         # first shape the mini-batch to have pixels in the rightmost dimension
#         x = x.reshape(-1, self.input_dim)
#         # then compute the hidden units
#         hidden = self.softplus(self.fc1(x))
#         # then return a mean vector and a (positive) square root covariance
#         # each of size batch_size x z_dim
#         z_loc = self.fc21(hidden)
#         z_scale = torch.exp(self.fc22(hidden))
#         return z_loc, z_scale

# class VAE(nn.Module):
#     # by default our latent space is 50-dimensional
#     # and we use 32 hidden units
#     def __init__(self, obs_dim, z_dim=2, hidden_dim=32, use_cuda=False):
#         super().__init__()
#         # create the encoder and decoder networks
#         self.encoder = Encoder(z_dim, hidden_dim, obs_dim)
#         self.decoder = Decoder(z_dim, hidden_dim, obs_dim)

#         if use_cuda:
#             # calling cuda() here will put all the parameters of
#             # the encoder and decoder networks into gpu memory
#             self.cuda()
#         self.use_cuda = use_cuda
#         self.z_dim = z_dim

#     # define the model p(x|z)p(z)
#     def model(self, x):
#         # register PyTorch module `decoder` with Pyro
#         pyro.module("decoder", self.decoder)
#         with pyro.plate("data", x.shape[0]):
#             # setup hyperparameters for prior p(z)
#             z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
#             z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
#             # sample from prior (value will be sampled by guide when computing the ELBO)
#             z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#             # decode the latent code z
#             loc_img = self.decoder(z)
#             # score against actual images
#             pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, x.shape[1]))

#     # define the guide (i.e. variational distribution) q(z|x)
#     def guide(self, x):
#         # register PyTorch module `encoder` with Pyro
#         pyro.module("encoder", self.encoder)
#         with pyro.plate("data", x.shape[0]):
#             # use the encoder to get the parameters used to define q(z|x)
#             z_loc, z_scale = self.encoder(x)
#             # sample the latent code z
#             pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

#     # define a helper function for reconstructing images
#     def reconstruct_img(self, x):
#         # encode image x
#         z_loc, z_scale = self.encoder(x)
#         # sample in latent space
#         z = dist.Normal(z_loc, z_scale).sample()
#         # decode the image (note we don't sample in image space)
#         loc_img = self.decoder(z)
#         return loc_img

# class MyVAE:
#     def __init__(self, n_components=2, hidden_dim=40) -> None:
#         self.n_components = n_components
#         self.hidden_dim = hidden_dim

#     def train(self, x, svi, train_loader, use_cuda=False):
#         # initialize loss accumulator
#         epoch_loss = 0.
#         # do a training epoch over each mini-batch x returned
#         # by the data loader
#         for x, _ in train_loader:
#             # if on GPU put mini-batch into CUDA memory
#             if use_cuda:
#                 x = x.cuda()
#             # do ELBO gradient and accumulate loss
#             epoch_loss += svi.step(x)

#         # return epoch loss
#         normalizer_train = len(train_loader.dataset)
#         total_epoch_loss_train = epoch_loss / normalizer_train
#         return total_epoch_loss_train
    
#     def evaluate(self, x, svi, test_loader, use_cuda=False):
#         # initialize loss accumulator
#         test_loss = 0.
#         # compute the loss over the entire test set
#         for x, _ in test_loader:
#             # if on GPU put mini-batch into CUDA memory
#             if use_cuda:
#                 x = x.cuda()
#             # compute ELBO estimate and accumulate loss
#             test_loss += svi.evaluate_loss(x)
#         normalizer_test = len(test_loader.dataset)
#         total_epoch_loss_test = test_loss / normalizer_test
#         return total_epoch_loss_test
    
#     def fit_transform(self, x):
#         pyro.set_rng_seed(0)
#         dataset = TensorDataset(torch.from_numpy(x).to(torch.float32), torch.zeros(x.shape[0]))
#         train_loader = DataLoader(dataset, batch_size=x.shape[0])
        
#         # Run options
#         LEARNING_RATE = 1.0e-3
#         USE_CUDA = False
        
#         # Enable smoke test - run the notebook cells on CI.
#         smoke_test = 'CI' in os.environ
#         # Run only for a single iteration for testing
#         NUM_EPOCHS = 1 if smoke_test else 100
#         # clear param store
#         pyro.clear_param_store()

#         # setup the VAE
#         vae = VAE(x.shape[1], use_cuda=USE_CUDA)

#         # setup the optimizer
#         adam_args = {"lr": LEARNING_RATE}
#         optimizer = Adam(adam_args)

#         # setup the inference algorithm
#         svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

#         train_elbo = []
#         test_elbo = []
#         # training loop
#         for epoch in range(NUM_EPOCHS):
#             total_epoch_loss_train = self.train(x, svi, train_loader, use_cuda=USE_CUDA)
#             train_elbo.append(-total_epoch_loss_train)
#             # print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

#         with torch.no_grad():
#             x, __ = next(iter(train_loader))
#             z_loc, z_scale = vae.encoder(x)
#             z = z_loc.detach().cpu().numpy()
#             return z



import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


class MyVAE:
    def __init__(self, n_components=2, epochs=10, batch_size=64, learning_rate=1e-3, hidden_dim=256, seed=0) -> None:
        self.n_components = n_components
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.seed = seed


    def fit_transform(self, raw_x):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        obs_dim = raw_x.shape[1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = TensorDataset(torch.from_numpy(raw_x).to(torch.float32), torch.zeros(raw_x.shape[0]))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        def loss_fn(recon_x, x, mean, log_var):
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(-1, obs_dim), x.view(-1, obs_dim), reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            return (BCE + KLD) / x.size(0)

        vae = VAE(
            encoder_layer_sizes=[obs_dim, self.hidden_dim],
            latent_size=self.n_components,
            decoder_layer_sizes=[self.hidden_dim, obs_dim]).to(device)

        optimizer = torch.optim.Adam(vae.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):

            for iteration, (x, y) in enumerate(data_loader):

                x, y = x.to(device), y.to(device)

                recon_x, mean, log_var, z = vae(x)

                loss = loss_fn(recon_x, x, mean, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            z = vae(torch.from_numpy(raw_x).to(torch.float32))[1].detach().cpu().numpy()
            return z