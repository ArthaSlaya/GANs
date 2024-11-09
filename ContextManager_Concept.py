import threading
import requests
import time
import psycopg2
import socket
import logging
import tempfile
import os
import mmap
from selenium import webdriver
import boto3
import zipfile
from cryptography.fernet import Fernet

# 1. File Manager
class FileManager:
    """
    A context manager for handling file operations.
    """
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

with FileManager("./Files/test.txt", "w") as f:
    f.write("Hello World")

# 2. Database Connection
class DatabaseManager:
    """
    A context manager for managing PostgreSQL database connections.
    """
    def __init__(self, dsn):
        self.dsn = dsn

    def __enter__(self):
        self.connection = psycopg2.connect(self.dsn)
        self.cursor = self.connection.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_value, traceback):
        self.cursor.close()
        self.connection.close()

with DatabaseManager("dbname=test user=postgres password=secret") as cursor:
    cursor.execute("SELECT * FROM my_table")

# 3. API Session Manager
class APISessionManager:
    """
    A context manager for managing API sessions using requests library.
    """
    def __enter__(self):
        self.session = requests.Session()
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

with APISessionManager() as session:
    response = session.get("https://api.example.com/data")

# 4. Thread Lock Manager
lock = threading.Lock()
class LockManager:
    """
    A context manager for acquiring and releasing a threading lock.
    """
    def __enter__(self):
        lock.acquire()
        print("Acquired lock")

    def __exit__(self, exc_type, exc_value, traceback):
        lock.release()
        print("Released lock")

with LockManager():
    print("Critical section")

# 5. Timer Context
class Timer:
    """
    A context manager to measure the execution time of a code block.
    """
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Elapsed time: {time.time() - self.start_time} seconds")

with Timer() as timer:
    time.sleep(2)

# 6. Socket Connection
class SocketManager:
    """
    A context manager for managing socket connections.
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        return self.sock

    def __exit__(self, exc_type, exc_value, traceback):
        self.sock.close()

with SocketManager("localhost", 8080) as sock:
    sock.sendall(b"Hello, World!")

# 7. Transaction Manager
class TransactionManager:
    """
    A context manager for managing database transactions.
    """
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        self.connection.autocommit = False

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.autocommit = True

with DatabaseManager("dbname=test user=postgres password=secret") as cursor:
    with TransactionManager(cursor.connection):
        cursor.execute("INSERT INTO my_table (name) VALUES ('test')")

# 8. Logging Manager
class LogManager:
    """
    A context manager for managing logging.
    """
    def __init__(self, logfile):
        self.logfile = logfile

    def __enter__(self):
        logging.basicConfig(filename=self.logfile, level=logging.INFO)
        logging.info("Entering the context")

    def __exit__(self, exc_type, exc_value, traceback):
        logging.info("Exiting the context")

with LogManager("./Files/app.log"):
    logging.info("Inside the context")

# 9. Temporary File Manager
class TempFileManager:
    """
    A context manager for creating and handling temporary files.
    """
    def __enter__(self):
        self.tmpfile = tempfile.NamedTemporaryFile(delete=False)
        return self.tmpfile

    def __exit__(self, exc_type, exc_value, traceback):
        os.unlink(self.tmpfile.name)

with TempFileManager() as tmp:
    tmp.write(b"Temporary data")

# 10. Network Connection Validator
class NetworkConnectionValidator:
    """
    A context manager for validating network connections.
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __enter__(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        return self.sock

    def __exit__(self, exc_type, exc_value, traceback):
        self.sock.close()

with NetworkConnectionValidator("www.google.com", 80) as conn:
    print("Connection established")

# 11. Semaphore Manager
semaphore = threading.Semaphore(2)
class SemaphoreManager:
    """
    A context manager for acquiring and releasing a threading semaphore.
    """
    def __enter__(self):
        semaphore.acquire()
        print("Semaphore acquired")

    def __exit__(self, exc_type, exc_value, traceback):
        semaphore.release()
        print("Semaphore released")

with SemaphoreManager():
    print("Inside semaphore-protected section")

# 12. WebDriver Manager
class WebDriverManager:
    """
    A context manager for managing Selenium WebDriver sessions.
    """
    def __enter__(self):
        self.driver = webdriver.Chrome()
        return self.driver

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.quit()

with WebDriverManager() as driver:
    driver.get("https://www.example.com")

# 13. Memory Map Manager
class MemoryMapManager:
    """
    A context manager for managing memory-mapped files.
    """
    def __init__(self, filename, size):
        self.filename = filename
        self.size = size

    def __enter__(self):
        self.file = open(self.filename, "r+b")
        self.mapped_file = mmap.mmap(self.file.fileno(), self.size)
        return self.mapped_file

    def __exit__(self, exc_type, exc_value, traceback):
        self.mapped_file.close()
        self.file.close()

with MemoryMapManager("./Files/testfile", 1024) as mmap_obj:
    print(mmap_obj[:])

# 14. OpenGL Resource Manager
class OpenGLResourceManager:
    """
    A context manager for managing OpenGL resources.
    """
    def __enter__(self):
        print("Allocating OpenGL resource")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Releasing OpenGL resource")

with OpenGLResourceManager():
    print("Using OpenGL resource")

# 15. GPU Resource Manager
class GpuResourceManager:
    """
    A context manager for managing GPU resources.
    """
    def __enter__(self):
        print("Allocating GPU resource")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Releasing GPU resource")

with GpuResourceManager():
    print("Using GPU resource")

# 16. File Compression Manager
class FileCompressionManager:
    """
    A context manager for compressing files using zip format.
    """
    def __init__(self, filepath, mode):
        self.filepath = filepath
        self.mode = mode

    def __enter__(self):
        self.zipfile = zipfile.ZipFile(self.filepath, self.mode)
        return self.zipfile

    def __exit__(self, exc_type, exc_value, traceback):
        self.zipfile.close()

with FileCompressionManager("./Files/example.zip", "w") as zf:
    zf.writestr("test.txt", "This is a test file")

# 17. AWS S3 Manager
class S3Manager:
    """
    A context manager for managing AWS S3 resource connections.
    """
    def __enter__(self):
        self.s3 = boto3.resource('s3')
        return self.s3

    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing S3 connection")

with S3Manager() as s3:
    for bucket in s3.buckets.all():
        print(bucket.name)

# 18. Cryptographic Key Manager
class CryptographicKeyManager:
    """
    A context manager for managing cryptographic keys.
    """
    def __init__(self, key):
        self.key = key

    def __enter__(self):
        self.cipher = Fernet(self.key)
        return self.cipher

    def __exit__(self, exc_type, exc_value, traceback):
        print("Destroying cryptographic key")

key = Fernet.generate_key()
with CryptographicKeyManager(key) as cipher:
    encrypted = cipher.encrypt(b"Secret data")
    print(f"Encrypted data: {encrypted}")

# 19. ZIP File Extract Manager
class ZipExtractManager:
    """
    A context manager for extracting zip files.
    """
    def __init__(self, zip_path, extract_path):
        self.zip_path = zip_path
        self.extract_path = extract_path

    def __enter__(self):
        self.zipfile = zipfile.ZipFile(self.zip_path, 'r')
        self.zipfile.extractall(self.extract_path)
        return self.zipfile

    def __exit__(self, exc_type, exc_value, traceback):
        self.zipfile.close()
        print("Zip file extracted and closed")

with ZipExtractManager("./Files/example.zip", "./Files/extracted/"):
    print("Zip file extracted successfully")

# 20. Temporary Directory Manager
class TempDirectoryManager:
    """
    A context manager for creating and managing temporary directories.
    """
    def __enter__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        return self.tempdir.name

    def __exit__(self, exc_type, exc_value, traceback):
        self.tempdir.cleanup()
        print("Temporary directory removed")

with TempDirectoryManager() as tempdir:
    print(f"Temporary directory created at: {tempdir}")

'''
DatabaseManager & TransactionManager:
    Ensure that the psycopg2 database credentials (dbname, user, password) match your database environment, or else the connection will fail.

WebDriverManager:
    The WebDriver setup (webdriver.Chrome()) requires that the ChromeDriver executable is properly installed and accessible in your system's PATH. 
    You may need to adjust this part depending on your setup.

AWS S3 Manager:
    This requires appropriate AWS credentials configured on your system to access S3. 
    Make sure that you have set up AWS credentials properly using the aws configure command or through environment variables.

Network Operations:
    Classes involving networking, such as SocketManager, NetworkConnectionValidator, and SemaphoreManager, assume that the server is running and available at the specified host and port. 
    The examples will fail if the connection cannot be made.

Cryptographic Key Manager:
    Make sure you have installed the cryptography library (pip install cryptography) if it is not already present.

LockManager and SemaphoreManager:
    The thread-related context managers (LockManager, SemaphoreManager) might not show significant behavior unless you run them in a multi-threaded context.

Directory Paths:
Ensure the ./Files/ directory and the files within it exist, or you may need to create them before running the script.

'''
