from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2D, LeakyReLU
import tensorflow as tf
import numpy as np

class Generator(Model):
    """
    A generator model for GAN that inherits from tf.keras.Model.
    This class defines a neural network that transforms a latent vector
    (random noise) into a generated image, using a series of dense,
    upsampling, and convolutional layers.

    Attributes
    ----------
    latent_dim : int
        Dimensionality of the latent space, which defines the size of the random noise vector.

    Methods
    -------
    call(inputs)
        Executes the forward pass of the generator model.
    
    generate_images(num_images=1)
        Generates a specified number of synthetic images by sampling random noise.
    """
    def __init__(self, latent_dim = 128):
        """
        Initializes the Generator with a specified latent dimension, defining
        the structure of the model with dense, upsampling, and convolutional layers.

        Parameters
        ----------
        latent_dim : int, optional
            Dimensionality of the latent space (default is 128).
        """

        # Why super(Generator, self) is Used?
        #     In the statement super(Generator, self).__init__():

        #     Generator specifies the current class for which we want to access the superclass method.
        #     self is the instance that’s calling super(), providing context for which method to call in the superclass.
        #     This syntax is especially useful in more complex inheritance chains, where multiple classes inherit from each other. 
        #     Using super(ClassName, self).__init__() ensures that Python’s method resolution order (MRO) is followed correctly, even if you add more classes into the hierarchy later.
        # Simplified Syntax in Python 3
        #     In Python 3, you can simplify this by omitting the explicit class and self, writing it as super().__init__(). 
        #     This accomplishes the same goal

        super(Generator, self).__init__()
        self.latent_dim = latent_dim # Latent space dimension (size of random noise vector)

        # Define the layers in the generator network
        self.dense = Dense(7 * 7 * 128, input_dim = self.latent_dim)
        self.leaky_relu_dense = LeakyReLU(0.2)
        self.reshape = Reshape((7, 7, 128))

        # Upsampling block 1 
        self.upsample1 = UpSampling2D()
        self.conv1 = Conv2D(128, kernel_size = 5, padding = 'same')
        self.leaky_relu1 = LeakyReLU(0.2)

        # Upsampling block 2 
        self.upsample2 = UpSampling2D()
        self.conv2 = Conv2D(128, kernel_size = 5, padding = 'same')
        self.leaky_relu2 = LeakyReLU(0.2)

        # Convolutional block 1
        self.conv3 = Conv2D(128, kernel_size = 4, padding = 'same')
        self.leaku_relu3 = LeakyReLU(0.2)

        # Convolution block 2
        self.conv4 = Conv2D(128, kernel_size = 4, padding = 'same')
        self.leaky_relu4 = LeakyReLU(0.2)

        # Final Convolution Block
        self.output_conv = Conv2D(1, kernel_size = 4, padding = 'same', activation = 'sigmoid')

    def call(self, inputs):
        """
        Defines the forward pass of the generator model.
        This method is called automatically when the model is used to generate images.

        Parameters
        ----------
        inputs : tf.Tensor
            A batch of latent vectors (random noise) with shape (batch_size, latent_dim).
        
        Returns
        -------
        tf.Tensor
            A batch of generated images with shape (batch_size, 28, 28, 1).
        """

        # Pass through the dense layer and apply LeakyReLU, then reshape
        x = self.dense(inputs)
        x = self.leaky_relu_dense(x)
        x = self.reshape(x)

        # First upsampling block
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)

        # Second upsampling block
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)

        # First convolutional block
        x = self.conv3(x)
        x = self.leaku_relu3(x)

        # Second convolutional block
        x = self.conv4(x)
        x = self.leaky_relu4(x)

        # Output layer to generate the final image
        return self.output_conv(x)
    
    def generate_images(self, num_images = 1):
        """
        Generates a specified number of synthetic images from random noise.

        Parameters
        ----------
        num_images : int, optional
            The number of images to generate (default is 1).
        
        Returns
        -------
        tf.Tensor
            A batch of generated images with shape (num_images, 28, 28, 1).
        
        Example
        -------
        >>> gen = Generator(latent_dim=128)
        >>> images = gen.generate_images(num_images=4)
        >>> print(images.shape)
        (4, 28, 28, 1)
        """

        # Create a batch of random latent vectors (random noise)
        random_latent_vectors = np.random.normal(loc= 0, scale= 1, size= (num_images, self.latent_dim))

        # Pass latent vectors through the generator model (uses call())
        return self(random_latent_vectors)  # Calls the overridden call() method

# Example Usage

# Initialize the generator with a latent dimension of 128
gen = Generator(latent_dim=128)

# Generate a batch of 4 images
generated_images = gen.generate_images(num_images=4)

# Print the shape of the generated images to confirm (4, 28, 28, 1)
print(f"Generated images shape: {generated_images.shape}")