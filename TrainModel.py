from ganBase import BaseGAN
from Generator import Generator
from Discriminator import Discriminator
import os

class FashionGANTrainer(BaseGAN):
    """
    FashionGANTrainer is a subclass of BaseGAN, specifically designed for training a GAN on the Fashion MNIST dataset.

    Attributes:
    ----------
    latent_dim : int
        The size of the latent vector used as input for the generator.

    Methods:
    -------
    __init__(latent_dim=128):
        Initializes the FashionGANTrainer with a generator, discriminator, and latent dimension.
    """
    
    def __init__(self, latent_dim=128):
        """
        Initializes the FashionGANTrainer by creating instances of the generator and discriminator.

        Parameters:
        ----------
        latent_dim : int, optional
            The size of the latent space (default is 128).
        """
        # Create the generator and discriminator models
        generator = Generator(latent_dim).model
        discriminator = Discriminator().model
        
        # Initialize the BaseGAN with generator, discriminator, and latent dimension
        super(FashionGANTrainer, self).__init__(generator, discriminator, latent_dim)

    def load_models(self, generator_path, discriminator_path):
        """
        Load saved generator and discriminator models.
        
        Parameters:
        ----------
        generator_path : str
            Path to the saved generator model.
        discriminator_path : str
            Path to the saved discriminator model.
        """
        if os.path.exists(generator_path):
            self.generator.load_weights(generator_path)
            print(f"Loaded generator model from {generator_path}")
        else:
            print(f"No discriminator model found at {discriminator_path}, starting from scratch.")