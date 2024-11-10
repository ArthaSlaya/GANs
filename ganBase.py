import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model 

class BaseGAN(tf.keras.Model):
    """
    BaseGAN is a superclass for implementing Generative Adversarial Networks (GANs).

    Attributes:
    ----------
    generator : tf.keras.Model
        The generator model which takes random noise and generates synthetic images.
    discriminator : tf.keras.Model
        The discriminator model which distinguishes between real and fake images.
    latent_dim : int
        The size of the latent vector used as input for the generator.
    g_optimizer : tf.keras.optimizers.Optimizer
        The optimizer for training the generator.
    d_optimizer : tf.keras.optimizers.Optimizer
        The optimizer for training the discriminator.
    g_loss_fn : tf.keras.losses.Loss
        The loss function for the generator.
    d_loss_fn : tf.keras.losses.Loss
        The loss function for the discriminator.

    Methods:
    -------
    compile():
        Compiles the GAN model by configuring optimizers and loss functions.
    train_step(real_images):
        Performs one step of training for both the generator and the discriminator.
    """
    
    def __init__(self, generator, discriminator, *args, **kwargs):
        """
        Initializes the FashionGAN with a generator and discriminator.

        Parameters:
        ----------
        generator : tf.keras.Model
            A Keras model representing the generator.
        discriminator : tf.keras.Model
            A Keras model representing the discriminator.
        *args :
            Additional positional arguments for the base class.
        **kwargs :
            Additional keyword arguments for the base class.
        """
        # Pass through args and kwargs to base class 
        super(BaseGAN, self).__init__(*args, **kwargs)

        # Create attributes for generator and discriminator
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        """
        Compiles the FashionGAN model by configuring optimizers and loss functions.

        Parameters:
        ----------
        g_opt : tf.keras.optimizers.Optimizer
            The optimizer for the generator.
        d_opt : tf.keras.optimizers.Optimizer
            The optimizer for the discriminator.
        g_loss : tf.keras.losses.Loss
            The loss function for the generator.
        d_loss : tf.keras.losses.Loss
            The loss function for the discriminator.
        *args :
            Additional positional arguments for the base class compile method.
        **kwargs :
            Additional keyword arguments for the base class compile method.
        """
        
        # Compile with base class
        super(BaseGAN, self).compile(*args, **kwargs)

        # Create attributes for losses and optimizers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        """
        Performs a single training step for both the generator and the discriminator.

        Parameters:
        ----------
        batch : tf.Tensor
            A batch of real images used to train the discriminator.

        Returns:
        -------
        dict
            A dictionary containing the discriminator loss ('d_loss') and generator loss ('g_loss').
        """

        # Get the real images from the batch
        real_images = batch

        # Generate fake images using the generator
        fake_images = self.generator(tf.random.normal(shape=(128, 128, 1)), training= False)

        # tf.GradientTape() 
        #     records all computations involving differentiable operations.
        #     It computes gradients needed for updating model parameters.
        #     This is extremely useful for implementing custom training loops, like those needed in GANs, where the discriminator and generator have separate loss functions and training processes.

        # Train the discriminator
        with tf.GradientTape() as d_tape: 
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True)  # Output for real images
            yhat_fake = self.discriminator(fake_images, training=True)  # Output for fake images
            # Concatenate real and fake outputs
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            
            # Create labels for real (0) and fake (1) images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)
            
            # Add noise to the TRUE labels to make training more robust
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            
            # Calculate the discriminator loss using binary cross-entropy
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)
            
        # Apply gradients to update the discriminator weights
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables) 
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
        
        # Train the generator
        with tf.GradientTape() as g_tape: 
            # Generate fake images for training the generator
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)
                                        
            # Get the discriminator's prediction for the generated images
            predicted_labels = self.discriminator(gen_images, training=False)
                                        
            # Calculate the generator's loss (goal: make discriminator classify fake images as real)
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels) 
            
        # Apply gradients to update the generator weights
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
        
        # Return both discriminator and generator losses
        return {"d_loss": total_d_loss, "g_loss": total_g_loss}