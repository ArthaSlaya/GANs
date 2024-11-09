# import tensorflow as tf
# from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import Model 

# class BaseGAN(tf.keras.Model):
#     """
#     A class representing a Generative Adversarial Network (GAN) for fashion dataset.

#     Attributes:
#     generator: The generator model used to create fake images.
#     discriminator: The discriminator model used to distinguish real from fake images.
#     g_opt: Optimizer used for training the generator.
#     d_opt: Optimizer used for training the discriminator.
#     g_loss: Loss function for the generator.
#     d_loss: Loss function for the discriminator.
#     """
    
#     def __init__(self, generator, discriminator, latent_dim=128):
#         """
#         Initializes the FashionGAN model with the generator and discriminator.

#         Args:
#         generator: A Keras model representing the generator.
#         discriminator: A Keras model representing the discriminator.
#         *args, **kwargs: Additional arguments to be passed to the base class.
#         """
#         # Pass through args and kwargs to base class 
#         super(BaseGAN, self).__init__(*args, **kwargs)

#         # Create attributes for generator and discriminator
#         self.generator = generator
#         self.discriminator = discriminator

#     def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
#         """
#         Compiles the GAN model with the specified optimizers and loss functions.

#         Args:
#         g_opt: Optimizer for the generator.
#         d_opt: Optimizer for the discriminator.
#         g_loss: Loss function for the generator.
#         d_loss: Loss function for the discriminator.
#         *args, **kwargs: Additional arguments to be passed to the base class.
#         """
        
#         # Compile with base class
#         super(BaseGAN, self).compile(*args, **kwargs)

#         # Create attributes for losses and optimizers
#         self.g_opt = g_opt
#         self.d_opt = d_opt
#         self.g_loss = g_loss
#         self.d_loss = d_loss

#     def train_step(self, batch):
#         """
#         Performs a single training step for the GAN, including updates for both the generator and discriminator.

#         Args:
#         batch: A batch of real images used for training the discriminator.

#         Returns:
#         A dictionary containing the discriminator loss ("d_loss") and generator loss ("g_loss").
#         """
#         # Get the real images from the batch
#         real_images = batch

#         # Generate fake images using the generator
#         fake_images = self.generator(tf.random.normal(shape=(128, 128, 1)), training= False)

#         # Train the discriminator
#         with tf.GradientTape() as d_tape:
#             # Pass the real and fake images to the discriminator model
#             yhat_real = self.discriminator(real_images, training = True)
#             yhat_fake = self.discriminator(fake_images, training = True)

#             # Concatenate the real and fake predictions
#             yhat_realfake = tf.concat([yhat_real, yhat_fake], axis= 0)

#             # Create labels for real (0) and fake (1) images
#             y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis= 0)

class Manager:
    def __init__(self):
        print("This is Init method")
    def __enter__(self):
        print("Entering the context")
        # Setup code (e.g., acquiring a resource)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")
        # Teardown code (e.g., releasing a resource)

# Usage
with Manager() as cm:
    print(cm)
    print("Inside the context")