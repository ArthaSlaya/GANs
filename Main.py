from DataPipeline import DataPipeline  # Import DataPipeline class for data loading and preprocessing
from TrainModel import FashionGANTrainer  # Import FashionGANTrainer class for model training
from Callbacks import ModelMonitor  # Import ModelMonitor class to visualize generated images during training

# Set up data pipeline
data_pipeline = DataPipeline()  # Instantiate the data pipeline for loading the Fashion MNIST dataset
dataset = data_pipeline.load_data()  # Load the dataset with preprocessing (normalization, batching, etc.)

# Initialize and compile the FashionGAN model
fashion_gan = FashionGANTrainer()  # Create an instance of the FashionGAN trainer
fashion_gan.compile()  # Compile the model with its optimizers and loss functions

# Start training with the ModelMonitor callback
fashion_gan.fit(dataset, epochs=20, callbacks=[ModelMonitor()])  # Train the model for 20 epochs with callback to save generated images