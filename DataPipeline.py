import tensorflow_datasets as tfds
import tensorflow as tf

class DataPipeline:
    """
    DataPipeline is a class that manages the loading and preprocessing of the Fashion MNIST dataset.

    Attributes:
    ----------
    batch_size : int
        The number of samples per batch of data.
    buffer_size : int
        The size of the buffer used for shuffling the dataset.

    Methods:
    -------
    load_data():
        Loads, preprocesses, and returns the Fashion MNIST dataset as a tf.data.Dataset.
    """
    
    def __init__(self, batch_size=128, buffer_size=60000):
        """
        Initializes the DataPipeline with batch size and buffer size for shuffling.

        Parameters:
        ----------
        batch_size : int, optional
            The number of samples per batch of data (default is 128).
        buffer_size : int, optional
            The size of the buffer used for shuffling the dataset (default is 60000).
        """
        self.batch_size = batch_size  # Set the batch size for batching the dataset
        self.buffer_size = buffer_size  # Set the buffer size for shuffling the dataset

    def load_data(self):
        """
        Loads the Fashion MNIST dataset, applies preprocessing steps, and returns it as a tf.data.Dataset.

        The preprocessing steps include:
        - Normalizing the image data to be in the range [0, 1]
        - Caching the dataset for improved performance
        - Shuffling the dataset with a specified buffer size
        - Batching the dataset into batches of the specified size
        - Prefetching to improve the efficiency of data loading

        Returns:
        -------
        tf.data.Dataset
            The preprocessed dataset ready for training.
        """
        # Load the Fashion MNIST dataset, specifically the 'train' split
        ds = tfds.load('fashion_mnist', split='train')
        
        # Normalize the image data to be in the range [0, 1]
        ds = ds.map(lambda x: x['image'] / 255.0)
        
        # Cache the dataset in memory for faster access
        ds = ds.cache()
        
        # Shuffle the dataset with the specified buffer size
        ds = ds.shuffle(self.buffer_size)
        
        # Batch the dataset with the specified batch size
        ds = ds.batch(self.batch_size)
        
        # Prefetch to improve data loading efficiency
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
