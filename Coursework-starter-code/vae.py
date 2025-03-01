from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


def hello_vae():
    print("Hello from vae.py!")

#  defines the VAE class with the encoder architecture
class VAE(nn.Module):
    def __init__(self, input_size,latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim =None# H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ###########################################################################
        # TODO: Implement the fully-connected encoder architecture described in   #
        # the notebook. Specifically, self.encoder should be a network that       #
        # inputs a batch of input images of shape (N, 1, H, W) into a batch of    #
        # hidden features of shape (N, H_d). Set up self.mu_layer and             #
        # self.logvar_layer to be a pair of linear layers that map the hidden     #
        # features into estimates of the mean and log-variance of the posterior   #
        # over the latent vectors; the mean and log-variance estimates will both  #
        # be tensors of shape (N, Z).                                             #
        ##########################################################################
        self.hidden_dim=400
        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Flatten(),# Flatten the input
            nn.Linear(self.input_size, self.hidden_dim),# Fully connected layer 1
            nn.ReLU(),# ReLU activation
            nn.Linear( self.hidden_dim, self.hidden_dim),# Fully connected layer 2
            nn.ReLU(),# ReLU activation
            nn.Linear( self.hidden_dim, self.hidden_dim),# Fully connected layer 3
            nn.ReLU()# ReLU activation
        )
        
        # Mu layer and logvar layer
        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)
        ##########################################################################
        # TODO: Implement the fully-connected decoder architecture described in   
        # the notebook. Specifically, self.decoder should be a network that inputs#
        # a batch of latent vectors of shape (N, Z) and outputs a tensor of       #
        # estimated images of shape (N, 1, H, W).                                 #
        ###########################################################################
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_dim), # Fully connected layer 1
            nn.ReLU(),  # ReLU activation
            nn.Linear(self.hidden_dim, self.hidden_dim), # Fully connected layer 2
            nn.ReLU(),  # ReLU activation
            nn.Linear(self.hidden_dim, self.hidden_dim), # Fully connected layer 3
            nn.ReLU(),  # ReLU activation
            nn.Linear(self.hidden_dim, self.input_size), # Fully connected layer 4
            nn.Sigmoid(),  # Sigmoid to map values to [0, 1]
            nn.Unflatten(dim=1,unflattened_size= (1, 28, 28))  # Unflatten to image shape (1, 28, 28)
        )
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ##########################################################################

# This function performs the forward pass through the FC-VAE model by passing the input images through the encoder, applying the reparametrization trick, and then passing the resulting latent vector through the decoder to reconstruct the input images.

    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)

        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent
          space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z),
          with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ###########################################################################
        # TODO: Implement the forward pass by following these steps               #
        # (1) Pass the input batch through the encoder model to get posterior     #
        #     mu and logvariance                                                  #
        # (2) Reparametrize to compute  the latent vector z                       #
        # (3) Pass z through the decoder to resconstruct x                        #
        #########################################################################
        # Pass the input batch through the encoder model to get posterior mu and logvariance
        # encoded = self.encoder(x.view(-1, self.input_size))
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
    
        # Reparametrize to compute the latent vector z
        z = reparametrize(mu, logvar)
        
        # Pass z through the decoder to reconstruct x
        x_hat = self.decoder(z)
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################
        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size,hidden_dim=400, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # C
        self.hidden_dim = None # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ###########################################################################
        # TODO: Define a FC encoder as described in the notebook that transforms  #
        # the image--after flattening and now adding our one-hot class vector (N, #
        # H*W + C)--into a hidden_dimension (N, H_d) feature space, and a final   #
        # two layers that project that feature space to posterior mu and posterior#
        # log-variance estimates of the latent space (N, Z)                       #
        ######################################################
        self.hidden_dim=400
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size + self.num_classes,  self.hidden_dim),# Concatenate input image with one-hot class vector
            nn.ReLU(),
            nn.Linear( self.hidden_dim,  self.hidden_dim),
            nn.ReLU(),
            nn.Linear( self.hidden_dim,  self.hidden_dim),
            nn.ReLU()
        )


        ###########################################################################
        # TODO: Define a fully-connected decoder as described in the notebook that#
        # transforms the latent space (N, Z + C) to the estimated images of shape #
        # (N, 1, H, W).                                                           #
        ###########################################################################
       
        # Mu layer and logvar layer
        self.mu_layer = nn.Linear( self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear( self.hidden_dim, self.latent_size)

        # Define the decoder architecture
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size + self.num_classes,  self.hidden_dim),  # Concatenate latent vector with one-hot class vector
            nn.ReLU(),
            nn.Linear( self.hidden_dim,  self.hidden_dim),
            nn.ReLU(),
            nn.Linear( self.hidden_dim,  self.hidden_dim),
            nn.ReLU(),
            nn.Linear( self.hidden_dim,self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
        
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)

        Returns:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent
          space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with
          Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ###########################################################################
        # TODO: Implement the forward pass by following these steps               #
        # (1) Pass the concatenation of input batch and one hot vectors through   #
        #     the encoder model to get posterior mu and logvariance               #
        # (2) Reparametrize to compute the latent vector z                        #
        # (3) Pass concatenation of z and one hot vectors through the decoder to  #
        #     resconstruct x                                                      #
        ###########################################################################
         # Concatenate input batch and one hot vectors
        x_concat = torch.cat((x.view(-1, self.input_size), c), dim=1)

        # Pass the concatenated input through the encoder to get posterior mu and logvariance
        encoded = self.encoder(x_concat)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)

        # Reparametrize to compute the latent vector z
        z = reparametrize(mu, logvar)

        # Concatenate z and one hot vectors and pass through the decoder to reconstruct x
        z_concat = torch.cat((z, c), dim=1)
        x_hat = self.decoder(z_concat)
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################
        return x_hat, mu, logvar


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance
    using the reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with
    mean mu and standard deviation sigma, such that we can backpropagate from the
    z back to mu and sigma. We can achieve this by first sampling a random value
    epsilon from a standard Gaussian distribution with zero mean and unit variance,
    then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network,
    it helps to pass this function the log of the variance of the distribution from
    which to sample, rather than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns:
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a
      Gaussian with mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ###############################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and    #
    # scaling by posterior mu and sigma to estimate z                             #
    ##############################################################################
    # Reparametrize by initializing epsilon as a normal distribution and scaling by posterior mu and sigma to estimate z
    std = torch.sqrt(torch.exp(logvar))
    # print(std)
    eps = torch.randn_like(mu)#
    # print(eps)
    z= mu + eps * std
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to
    formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space
      dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z
      latent space dimension

    Returns:
    - loss: Tensor containing the scalar loss for the negative variational
      lowerbound
    """
    loss = None
    ###############################################################################
    # TODO: Compute negative variational lowerbound loss as described in the      #
    # notebook                                                                    #
    ##############################################################################     # Reconstruction loss using binary cross-entropy loss
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = recon_loss + kl_divergence
    
    # Average across samples in the minibatch
    loss /= x.size(0)
    ###############################################################################
    #                            END OF YOUR CODE                                 #
    ###############################################################################
    return loss
