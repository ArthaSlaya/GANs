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
                # First layer: Fully connected layer to reshape the random input vector
                Dense(7 * 7 * 128, input_dim = self.latent_dim), # Output shape: (7*7*128,)
                LeakyReLU(0.2),
                Reshape((7, 7, 128)), # Reshape to (7, 7, 128) for the Conv2D layers

                # Upsampling and convolutional layers to build up the image
                # Upsampling block 1 
                UpSampling2D(),
                Conv2D(128, kernel_size = 5, padding = 'same'),
                LeakyReLU(0.2),

                # Upsampling block 2 
                UpSampling2D(),
                Conv2D(128, kernel_size = 5, padding = 'same'),
                LeakyReLU(0.2),

                # Convolutional block 1
                Conv2D(128, kernel_size = 4, padding = 'same'),
                LeakyReLU(0.2),

                # Convolutional block 2
                Conv2D(128, kernel_size = 4, padding = 'same'),
                LeakyReLU(0.2),

                # Conv layer to get to one channel
                Conv2D(1, kernel_size = 4, padding = 'same', activation = 'sigmoid')
            ]
        )
        return model
    
#     def generate_images(self, num_images = 1):
#         """
#         Generates synthetic images by feeding random latent vectors through the generator model.

#         Parameters:
#         ----------
#         num_images : int, optional
#             The number of images to generate (default is 1).

#         Returns:
#         -------
#         numpy.ndarray
#             A batch of generated images with shape (num_images, 28, 28, 1).
#         """
#         # Generate random latent vectors for input
#         random_latent_vectors = np.random.normal(loc= 0, scale= 1, size= (num_images, self.latent_dim))

#         # Pass latent vectors through the generator model
#         generated_images = self.model.predict(random_latent_vectors)
#         return generated_images
    
# # Example Usage

# # Instantiate the generator
# gen = Generator(latent_dim=128)

# # Print a summary of the model architecture
# gen.model.summary()

# # Generate a batch of 4 synthetic images
# generated_images = gen.generate_images(num_images=4)

# # Check the shape of the generated images
# for i, img in enumerate(generated_images):
#     print(f"Generated image {i+1} shape: {img.shape}")