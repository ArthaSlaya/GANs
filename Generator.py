# Version Without Deriving from tf.keras.Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2D, LeakyReLU
import numpy as np

class Generator:
    """
    The Generator class builds a neural network model that generates synthetic 
    images from random noise (latent vectors). This is typically used as part 
    of a GAN (Generative Adversarial Network).

    Attributes:
    ----------
    latent_dim : int
        The dimensionality of the latent space. It defines the size of the random 
        noise vector used as input to generate images.

    model : tensorflow.keras.Model
        The Sequential Keras model representing the generator neural network.

    Methods:
    -------
    build_generator():
        Builds and returns the generator model using Keras layers.
    
    generate_images(num_images=1):
        Generates a specified number of synthetic images from random noise.
    """

    def __init__(self, latent_dim=128):
        """
        Initializes the Generator with a specified latent dimension, 
        and builds the generator model.

        Parameters:
        ----------
        latent_dim : int, optional
            Dimensionality of the latent space (default is 128).
        """
        self.latent_dim = latent_dim  # Size of the random noise vector
        self.model = self.build_generator()  # Build the generator model

    def build_generator(self):
        """
        Constructs the generator neural network model using a Sequential API.

        The network architecture:
        - Dense layer with 6272 units (7*7*128), followed by LeakyReLU activation
        - Reshape layer to format the output to (7, 7, 128) for the convolutional layers
        - Upsampling and Conv2D layers to increase image size
        - Final Conv2D layer to produce a single-channel output image with pixel values
          scaled by a sigmoid activation function

        Returns:
        -------
        tensorflow.keras.Model
            The generator model.
        """
        model = Sequential(
            [
                    Dense(7 * 7 * 128, input_dim = self.latent_dim),
                    LeakyReLU(0.2),
                    Reshape((7, 7, 128)),

                    
            ]
        )