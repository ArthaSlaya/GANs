import tensorflow_datasets as tfds
import tensorflow as tf

class DataPipeline:
    """
    A class for loading, preprocessing, and batching the Fashion MNIST dataset
    to prepare it for training. This class uses the TensorFlow Datasets (tfds)
    API to load the data and applies transformations for efficient processing.

    Attributes:
        batch_size (int): The number of samples per batch.
        buffer_size (int): The buffer size for shuffling the dataset.
    """
    def __init__(self, batch_size=128, buffer_size=60000):
        """
        Initializes the DataPipeline with specified batch and buffer sizes.

        Args:
            batch_size (int): The number of samples per batch. Default is 128.
            buffer_size (int): The buffer size for shuffling the dataset.
                               Default is 60000 (full dataset size for Fashion MNIST).
        """
        self.batch_size = batch_size

        # What is Buffer Size in Shuffling?
        #     When you shuffle data, a buffer is used to hold a subset of the dataset in memory, and then the items in this buffer are randomly mixed. 
        #     TensorFlow's tf.data.Dataset.shuffle(buffer_size) method utilizes this buffer to improve shuffling performance while balancing memory use and randomness.

        # How Buffer Size Affects Shuffling
        # Small Buffer Size:
        #     If the buffer size is smaller than the dataset (e.g., buffer_size=100 on a dataset with 60,000 samples), only a small portion of the dataset is shuffled at any time.
        #     This creates a limited randomization effect within the buffer and leads to less effective shuffling for larger datasets, as samples may maintain a partial ordering.
            
        # Large Buffer Size (e.g., Equal to the Dataset Size):
        #     Setting the buffer size to the dataset size (e.g., buffer_size=60000 for Fashion MNIST) ensures that all samples have an equal chance of appearing at any position in the shuffled dataset.
        #     This achieves complete shuffling but may consume a lot of memory, as it loads the entire dataset into memory at once.
        #     For smaller datasets (like Fashion MNIST), this is often practical and gives the best shuffling quality.

        # Very Large Datasets:
        #     For very large datasets that don’t fit into memory, setting the buffer size to the full dataset size is impractical.
        #     Instead, you can set a reasonable buffer size based on your system's memory capacity. For example, if you have 16 GB of memory, you might set a buffer size that fits comfortably within this limit.
        #     Choosing a smaller buffer size (e.g., buffer_size=10000 on a dataset with millions of samples) still provides some degree of randomization while being memory-efficient.
            
        self.buffer_size = buffer_size

    def load_data(self):
        """ 
        Loads and preprocesses the Fashion MNIST dataset, including scaling, caching, shuffling, batching, and prefetching.

        Returns:
            tf.data.Dataset: A preprocessed dataset ready for model training.
        """

        # Load the Fashion MNIST dataset split for training
        # The Fashion MNIST dataset is typically pre-split into training and testing sets, so you can load either one individually or both together.
        # In TensorFlow Datasets (tfds), specifying split='train' means you're requesting only the training portion of the dataset.

        # Here's a breakdown:

        # 1. split='train':
        #     This loads only the training portion of the dataset (typically 60,000 images for Fashion MNIST).

        # 2. split='test':
        #     This loads only the testing portion of the dataset (10,000 images for Fashion MNIST).

        # 3. Combining Splits:
        #     You can load both training and testing data together by specifying multiple splits:
        #         ds_train, ds_test = tfds.load('fashion_mnist', split=['train', 'test'])
        #     This provides two datasets: one for training and one for testing.

        # 4. Partial Splits:
        #     You can also load a subset of the training or test data, e.g., split='train[:80%]' would load the first 80% of the training data, and split='train[80%:]' would load the remaining 20%.
        #     1. Load the First 50% of the Test Data:
        #         ds_test = tfds.load('fashion_mnist', split='test[:50%]')
        #         This loads the first 50% of the test data (5,000 samples if Fashion MNIST has 10,000 test samples).

        #     2. Load the Last 30% of the Test Data:
        #         ds_test = tfds.load('fashion_mnist', split='test[-30%:]')
        #         This loads the last 30% of the test data (3,000 samples for Fashion MNIST).
            
        #     3. Split Test Data into 3 Equal Parts:
        #         ds_test1 = tfds.load('fashion_mnist', split='test[:33%]')
        #         ds_test2 = tfds.load('fashion_mnist', split='test[33%:66%]')
        #         ds_test3 = tfds.load('fashion_mnist', split='test[66%:]')
        #         This loads three separate subsets, each covering about one-third of the test data.

        #     4. Load Specific Number of Samples from Test Data:
        #         ds_test = tfds.load('fashion_mnist', split='test[:1000]')
        #         This loads only the first 1,000 samples of the test data.

        #     5. Combining Partial Splits:
        #         You can also load multiple partial splits at once, for example, by combining parts of the test data with parts of the training data
        #         ds_combined = tfds.load('fashion_mnist', split=['train[:50%]', 'test[:50%]'])

        ds = tfds.load('fashion_mnist', split='train')

        # Normalize image pixel values from [0, 255] to [0, 1]
        # num_parallel_calls=tf.data.AUTOTUNE enables parallel processing for faster performance.
        ds = ds.map(lambda x: x['image']/255.0, num_parallel_calls=tf.data.AUTOTUNE)

        

