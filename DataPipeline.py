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

        # The `num_parallel_calls=tf.data.AUTOTUNE` parameter in TensorFlow controls
        # the number of parallel CPU threads used to perform operations in the data pipeline.

        # When using `map` to apply a transformation to each element in the dataset,
        # setting `num_parallel_calls` enables these transformations to happen in parallel,
        # rather than one at a time. This is particularly useful for large datasets,
        # as it speeds up the data processing pipeline.

        # Setting `num_parallel_calls=tf.data.AUTOTUNE`:
        # - This instructs TensorFlow to automatically determine the optimal number of parallel CPU threads based on available system resources (like the number of CPU cores).
        # - The `AUTOTUNE` option dynamically adjusts the number of parallel calls to optimize performance, eliminating the need for manual tuning.

        # Benefits of `num_parallel_calls=tf.data.AUTOTUNE`:
        # 1. Faster Data Processing: By parallelizing transformations, it reduces bottlenecks and allows data to be fed to the model at a faster rate.
        # 2. Automatic Tuning: TensorFlow chooses the optimal number of parallel calls based on the system’s capabilities, which can vary depending on the hardware (e.g., CPU cores).
        # 3. Reduces Manual Optimization: Without `AUTOTUNE`, you would need to experiment with different values for `num_parallel_calls` to find the best setting. `AUTOTUNE` automates this.
        ds = ds.map(lambda x: tf.cast(x['image'], tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Caching and Memory Usage in TensorFlow
    
        # 1. Dataset Caching with .cache():
        #    - By default, .cache() in TensorFlow attempts to load the entire dataset into memory.
        #      For small datasets like Fashion MNIST (~28MB for 60,000 training samples), this allows for faster access in each epoch by avoiding reloading from disk.
        #    - For larger datasets, you can specify a file path (e.g., .cache('/path/to/cache_file')) to cache the data on disk rather than in memory.
        #      Disk-based caching is slower than in-memory caching but useful when working with very large datasets that exceed memory limits.
        #    - TensorFlow automatically handles caching based on available memory. If the dataset doesn’t fit in memory, caching may be skipped or partially cached without throwing an error.

        # 2. Pixel Data Representation in Fashion MNIST
        #    - The Fashion MNIST images are stored in 8-bit grayscale format:
        #        - Each pixel has a value ranging from 0 to 255, represented as an unsigned 8-bit integer (uint8), which requires only 1 byte per pixel.
        #        - This is an efficient representation, as uint8 is sufficient for grayscale values without excessive memory usage.

        # 3. Memory Consumption for 8-bit Grayscale Images
        #    - Single Pixel: Each pixel uses 1 byte (uint8) to store values between 0-255.
        #    - Single Image: A Fashion MNIST image is 28x28 pixels, totaling 784 pixels, requiring 784 bytes per image.
        #    - Entire Dataset: For 60,000 images, the memory requirement is approximately 60,000 * 784 bytes = ~47 MB.

        # 4. Why Avoid int32 for Grayscale Pixels?
        #    - Using int32 (4 bytes per pixel) would significantly increase memory usage without adding any benefit, as the 0-255 range can be fully represented by uint8.
        #    - For example, switching to int32 would increase the dataset size to around 188 MB (4 times more than uint8), which is inefficient for a dataset like Fashion MNIST.
        ds = ds.cache()

        # Shuffle the dataset to randomize the order of images
        # Shuffling with buffer size equal to the dataset size ensures good randomization.
        ds = ds.shuffle(self.buffer_size)

        # Batch the dataset to process multiple images at once
        # Batching helps in efficient computation by processing groups of images together.
        ds = ds.batch(self.batch_size)

        # Prefetch batches to improve data loading efficiency
        # 
        # Prefetching allows the data pipeline to load the next batch of data in the background
        # while the current batch is being processed by the model. This means:
        # 1. The model almost always has a batch ready to process, reducing waiting time.
        # 2. Training becomes faster and more efficient, as we eliminate idle time for the model.
        #
        # Example of prefetching:
        # - Without prefetch: The model completes processing batch 1, then waits for the pipeline
        #   to load and preprocess batch 2. This idle waiting time adds up over multiple batches.
        # - With prefetch: While the model is processing batch 1, the data pipeline is already
        #   preparing batch 2 in the background. When the model is ready for batch 2, it’s
        #   instantly available. This overlapping of loading and processing improves performance.
        #
        # `tf.data.AUTOTUNE` automatically sets an optimal buffer size for prefetching, making
        # this process adaptable to different system resources and load, without manual tuning.
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds