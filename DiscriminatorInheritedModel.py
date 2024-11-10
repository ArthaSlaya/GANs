# from tensorflow.keras import Model
# from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense
# import tensorflow as tf

# class Discriminator(Model):
#     """
#     Discriminator model for a GAN, implemented as a subclass of tf.keras.Model.
#     This model is designed to classify input images as real or fake by passing
#     them through a series of convolutional layers with LeakyReLU activations,
#     dropout layers for regularization, and a final dense layer for binary classification.

#     Attributes
#     ----------
#     conv1, conv2, conv3, conv4 : Conv2D
#         Convolutional layers for extracting features from images.
#     leakyrelu1, leakyrelu2, leakyrelu3, leakyrelu4 : LeakyReLU
#         Activation functions applied after each Conv2D layer to introduce non-linearity.
#     dropout1, dropout2, dropout3, dropout4 : Dropout
#         Dropout layers to prevent overfitting by randomly setting activations to zero.
#     flatten_output : Flatten
#         Flattens the feature maps before passing to the dense layer.
#     dropout_output : Dropout
#         Dropout applied after flattening for regularization.
#     dense_output : Dense
#         Final output layer with sigmoid activation for binary classification.
#     """

#     def __init__(self):
#         """
#         Initializes the Discriminator model and defines its layers.
#         """
#         super().__init__()
#         self.build_discriminator()  # Build the discriminator layers

#     def build_discriminator(self):
#         """
#         Defines the layers of the discriminator model.
#         """
#         # First convolutional block
#         self.conv1 = Conv2D(32, kernel_size=5, input_shape=(28, 28, 1))
#         self.leakyrelu1 = LeakyReLU(0.2)
#         self.dropout1 = Dropout(0.4)

#         # Second convolutional block
#         self.conv2 = Conv2D(64, kernel_size=5)
#         self.leakyrelu2 = LeakyReLU(0.2)
#         self.dropout2 = Dropout(0.4)

#         # Third convolutional block
#         self.conv3 = Conv2D(128, kernel_size=5)
#         self.leakyrelu3 = LeakyReLU(0.2)
#         self.dropout3 = Dropout(0.4)

#         # Fourth convolutional block
#         self.conv4 = Conv2D(256, kernel_size=5)
#         self.leakyrelu4 = LeakyReLU(0.2)
#         self.dropout4 = Dropout(0.4)

#         # Flatten and output layers
#         self.flatten_output = Flatten()
#         self.dropout_output = Dropout(0.4)
#         self.dense_output = Dense(1, activation='sigmoid')

#     def call(self, inputs):
#         """
#         Defines the forward pass of the discriminator model.
        
#         Parameters
#         ----------
#         inputs : tf.Tensor
#             A batch of images with shape (batch_size, 28, 28, 1).
        
#         Returns
#         -------
#         tf.Tensor
#             A tensor with shape (batch_size, 1), representing the probability
#             that each image is real (close to 1) or fake (close to 0).
#         """
#         # First convolutional block
#         x = self.conv1(inputs)
#         x = self.leakyrelu1(x)
#         x = self.dropout1(x)

#         # Second convolutional block
#         x = self.conv2(x)
#         x = self.leakyrelu2(x)
#         x = self.dropout2(x)

#         # Third convolutional block
#         x = self.conv3(x)
#         x = self.leakyrelu3(x)
#         x = self.dropout3(x)

#         # Fourth convolutional block
#         x = self.conv4(x)
#         x = self.leakyrelu4(x)
#         x = self.dropout4(x)

#         # Flatten and output layer
#         x = self.flatten_output(x)
#         x = self.dropout_output(x)
#         x = self.dense_output(x)

#         return x

# # # Example Usage

# # # Instantiate the discriminator
# # discriminator = Discriminator()

# # # Define a batch of random images (e.g., noise) with shape (4, 28, 28, 1)
# # random_images = tf.random.normal([4, 28, 28, 1])

# # # Pass the images through the discriminator to get predictions
# # predictions = discriminator(random_images)

# # # Print the output shape and values
# # print(f"Predictions shape: {predictions.shape}")
# # print(f"Predictions: {predictions.numpy()}")

# # Instantiate the discriminator
# discriminator = Discriminator()

# # Build the model by specifying the input shape
# discriminator.build(input_shape=(None, 28, 28, 1))

# # Print the model summary
# discriminator.summary()


from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense
import tensorflow as tf

class Discriminator(Model):
    """
    Discriminator model for a GAN, implemented as a subclass of tf.keras.Model.
    This model classifies input images as real or fake.
    """

    def __init__(self):
        """
        Initializes the Discriminator model by defining its layers.
        """
        super(Discriminator, self).__init__()
        
        # First convolutional block
        self.conv1 = Conv2D(32, kernel_size=5)
        self.leakyrelu1 = LeakyReLU(0.2)
        self.dropout1 = Dropout(0.4)

        # Second convolutional block
        self.conv2 = Conv2D(64, kernel_size=5)
        self.leakyrelu2 = LeakyReLU(0.2)
        self.dropout2 = Dropout(0.4)

        # Third convolutional block
        self.conv3 = Conv2D(128, kernel_size=5)
        self.leakyrelu3 = LeakyReLU(0.2)
        self.dropout3 = Dropout(0.4)

        # Fourth convolutional block
        self.conv4 = Conv2D(256, kernel_size=5)
        self.leakyrelu4 = LeakyReLU(0.2)
        self.dropout4 = Dropout(0.4)

        # Flatten and output layers
        self.flatten_output = Flatten()
        self.dropout_output = Dropout(0.4)
        self.dense_output = Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        Defines the forward pass of the discriminator model.
        
        Parameters
        ----------
        inputs : tf.Tensor
            A batch of images with shape (batch_size, 28, 28, 1).
        
        Returns
        -------
        tf.Tensor
            A tensor with shape (batch_size, 1), representing the probability
            that each image is real (close to 1) or fake (close to 0).
        """
        x = self.conv1(inputs)
        x = self.leakyrelu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.leakyrelu4(x)
        x = self.dropout4(x)

        x = self.flatten_output(x)
        x = self.dropout_output(x)
        return self.dense_output(x)

# Instantiate the discriminator
discriminator = Discriminator()

# Call the model with sample data to build it
sample_input = tf.random.normal([1, 28, 28, 1])
discriminator(sample_input)  # Run a forward pass with sample input

# Print the model summary
discriminator.summary()