import os
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow as tf

class ModelMonitor(Callback):
    """
    ModelMonitor is a custom Keras callback used to monitor the training of a GAN by saving generated images at the end of each epoch.

    Attributes:
    ----------
    num_images : int
        The number of images to generate at the end of each epoch.
    latent_dim : int
        The size of the latent vector used as input for the generator.
    save_dir : str
        The directory where the generated images will be saved.

    Methods:
    -------
    on_epoch_end(epoch, logs=None):
        Called at the end of each epoch to generate and save images.
    """
    
    def __init__(self, num_images=3, latent_dim=128, save_dir="images"):
        """
        Initializes the ModelMonitor callback with the number of images to generate, latent dimension, and save directory.

        Parameters:
        ----------
        num_images : int, optional
            The number of images to generate at the end of each epoch (default is 3).
        latent_dim : int, optional
            The size of the latent vector used as input for the generator (default is 128).
        save_dir : str, optional
            The directory where the generated images will be saved (default is "images").
        """
        self.num_images = num_images  # Set the number of images to generate at the end of each epoch
        self.latent_dim = latent_dim  # Set the size of the latent vector for the generator
        self.save_dir = save_dir  # Set the directory to save generated images
        # Create the save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to generate and save images using the generator model.

        Parameters:
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Contains the logs of the epoch, such as loss and metrics (default is None).
        """
        # Generate random latent vectors
        random_latent_vectors = tf.random.normal((self.num_images, self.latent_dim))
        # Generate images using the generator model
        generated_images = self.model.generator(random_latent_vectors)
        # Save each generated image to the specified directory
        for i, img in enumerate(generated_images):
            # Convert the generated image to a PIL image and scale pixel values
            img = array_to_img(img * 255.0)
            # Save the image with a filename that includes the epoch and image index
            img.save(f"{self.save_dir}/generated_img_{epoch}_{i}.png")
