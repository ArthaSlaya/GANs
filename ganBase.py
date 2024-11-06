import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

class BaseGAN(tf.keras.Model):
    """
    A class representing a basic Generative Adversarial Network (GAN).

    Attributes:
    generator: The generator model used to create fake images.
    discriminator: The discriminator model used to distinguish real from fake images.
    latent_dim: Dimension of the latent space vector used as input for the generator.
    g_optimizer: Optimizer used for training the generator.
    d_optimizer: Optimizer used for training the discriminator.
    g_loss_fn: Loss function for the generator.
    d_loss_fn: Loss function for the discriminator.
    """
    
    def __init__(self, generator, discriminator, latent_dim=128):
        """
        Initializes the GAN model with the generator and discriminator.

        Args:
        generator: A Keras model representing the generator.
        discriminator: A Keras model representing the discriminator.
        latent_dim: Integer specifying the dimension of the latent vector. Default is 128.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        # Initialize optimizers and loss functions for both generator and discriminator
        self.g_optimizer = Adam(learning_rate=0.0001)
        self.d_optimizer = Adam(learning_rate=0.00001)
        self.g_loss_fn = BinaryCrossentropy()
        self.d_loss_fn = BinaryCrossentropy()

    def compile(self):
        """
        Compiles the GAN model with the specified optimizers and loss functions.
        """
        # Call superclass's compile method
        super(BaseGAN, self).compile()
        # Set optimizers and loss functions
        self.g_optimizer = self.g_optimizer
        self.d_optimizer = self.d_optimizer
        self.g_loss_fn = self.g_loss_fn
        self.d_loss_fn = self.d_loss_fn

    def train_step(self, real_images):
        """
        Performs a single training step for the GAN, including updates for both the generator and discriminator.

        Args:
        real_images: A batch of real images used for training the discriminator.

        Returns:
        A dictionary containing the discriminator loss ("d_loss") and generator loss ("g_loss").
        """
        # Get the batch size from the input images
        batch_size = tf.shape(real_images)[0]
        # Generate random latent vectors
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Generate fake images using the generator
        generated_images = self.generator(random_latent_vectors)

        # Train the discriminator
        with tf.GradientTape() as tape:
            # Pass real and generated images through the discriminator
            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(generated_images)
            # Calculate discriminator loss for real images (target = 1) and fake images (target = 0)
            d_loss = self.d_loss_fn(tf.ones_like(real_output), real_output) + \
                     self.d_loss_fn(tf.zeros_like(fake_output), fake_output)

        # Compute gradients of the discriminator loss with respect to its trainable weights
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        # Apply gradients to update the discriminator weights
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train the generator
        with tf.GradientTape() as tape:
            # Generate new fake images
            generated_images = self.generator(random_latent_vectors)
            # Pass generated images through the discriminator
            fake_output = self.discriminator(generated_images)
            # Calculate generator loss (target for generated images = 1)
            g_loss = self.g_loss_fn(tf.ones_like(fake_output), fake_output)

        # Compute gradients of the generator loss with respect to its trainable weights
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        # Apply gradients to update the generator weights
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Return the losses for both the generator and discriminator
        return {"d_loss": d_loss, "g_loss": g_loss}

import tensorflow as tf
from tensorflow.keras import models, layers, Layer

class SimpleDense(Layer):

  def __init__(self, units=32):
      print('1.')
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):  # Create the state of the layer (weights)
    print('2.')
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'),
        trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True)

  def call(self, inputs):  # Defines the computation from inputs to outputs
      print('3.')
      return tf.matmul(inputs, self.w) + self.b

# Instantiates the layer.
linear_layer = SimpleDense(4)

# This will also call `build(input_shape)` and create the weights.
y = linear_layer(tf.ones((2, 2)))
# assert len(linear_layer.weights) == 2, f"linear layer weights : {linear_layer.weights}"

# These weights are trainable, so they're listed in `trainable_weights`:
# assert len(linear_layer.trainable_weights) == 2
print(F"trainable weights : {linear_layer.trainable_weights}")