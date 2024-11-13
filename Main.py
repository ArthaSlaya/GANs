from DataPipeline import DataPipeline  # Import DataPipeline class for data loading and preprocessing
from TrainModel import FashionGANTrainer  # Import FashionGANTrainer class for model training
from Callbacks import ModelMonitor  # Import ModelMonitor class to visualize generated images during training

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Set up data pipeline
data_pipeline = DataPipeline(batch_size=128)  # Instantiate the data pipeline for loading the Fashion MNIST dataset
dataset = data_pipeline.load_data()  # Load the dataset with preprocessing (normalization, batching, etc.)

# Initialize and compile the FashionGAN model
fashion_gan = FashionGANTrainer(latent_dim=128)  # Create an instance of the FashionGAN trainer

# Set up optimizers and loss functions
g_opt = Adam(learning_rate=0.0001)  # Generator optimizer
d_opt = Adam(learning_rate=0.00001)  # Discriminator optimizer
g_loss = BinaryCrossentropy()  # Generator loss function
d_loss = BinaryCrossentropy()  # Discriminator loss function

fashion_gan.compile(g_opt=g_opt, d_opt=d_opt, g_loss=g_loss, d_loss=d_loss)  # Compile the model with its optimizers and loss functions

# Start training with the ModelMonitor callback
fashion_gan.fit(dataset, epochs=20, callbacks=[ModelMonitor(num_images=4)])  # Train the model for 20 epochs with callback to save generated images