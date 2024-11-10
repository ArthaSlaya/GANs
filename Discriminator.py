from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense

class Discriminator:
    """
    Discriminator model for a GAN, implemented without inheriting from 
    tf.keras.Model. This class builds a convolutional neural network using 
    the Sequential API, designed to classify images as real or fake.

    Attributes
    ----------
    model : tf.keras.Model
        The Sequential Keras model representing the discriminator network.

    Methods
    -------
    build_discriminator()
        Constructs the discriminator model, returning a Keras Sequential model.
    """
    def __init__(self):
        """
        Initializes the Discriminator model by building the network structure
        and storing it in the model attribute.
        """
        self.model = self.build_discriminator()  # Build and store the Sequential model

    def build_discriminator(self):
        """
        Constructs the architecture of the discriminator model.

        The model structure:
        - Convolutional layers with LeakyReLU activation for extracting features
        - Dropout layers to reduce overfitting
        - A Flatten layer to convert feature maps to a 1D vector
        - A Dense output layer with a sigmoid activation for binary classification (real or fake)

        Returns
        -------
        tf.keras.Model
            The constructed Keras Sequential model for the discriminator.
        """
        model = Sequential(
            [
                # First convolutional block
                Conv2D(32, kernel_size = 5, input_shape = (28, 28, 1)), # Convolutional layer with 32 filters
                LeakyReLU(0.2),  # Leaky ReLU activation to introduce non-linearity
                Dropout(0.4), # Dropout to prevent overfitting

                # Second convolutional block
                Conv2D(64, kernel_size = 5),
                LeakyReLU(0.2),
                Dropout(0.4),

                # Third convolutional block
                Conv2D(128, kernel_size = 5),
                LeakyReLU(0.2),
                Dropout(0.4),

                # Fourth convolutional block    
                Conv2D(256, kernel_size = 5),
                LeakyReLU(0.2),
                Dropout(0.4),

                # Flatten and output layers
                Flatten(), # Flatten feature maps to a 1D vector for the dense layer
                Dropout(0.4), # Dropout for regularization before the output layer
                Dense(1, activation = 'sigmoid') # Sigmoid activation for binary classification (real/fake)
            ]
        )

        return model

# Instantiate the Discriminator
discriminator = Discriminator()

# Access the Sequential model
model = discriminator.model

# Print a summary of the model architecture
model.summary()