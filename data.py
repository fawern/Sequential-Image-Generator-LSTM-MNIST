"""
MNIST data
"""

import pandas as pd
from keras.datasets import mnist # type: ignore
from typing import Tuple

def get_mnist_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get MNIST data and return pandas DataFrame

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_data, test_data
    """
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(f"Train images shape : {train_images.shape}")
    print(f"Shape of single image : {train_images[0].shape}")

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    data = pd.DataFrame({
        "labels" : train_labels, 'imgs' : train_images.tolist()
    })

    return data
