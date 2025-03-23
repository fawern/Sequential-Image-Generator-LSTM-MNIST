"""
Test the model prediction
"""

import matplotlib.pyplot as plt
import numpy as np

from model import NextImagePredictor

def test_model_prediction(
    model: NextImagePredictor,
    X_test: np.ndarray,
    n: int
) -> None:
    """
    Generate the image from the model prediction

    Args:
        - model: NextImagePredictor
        - X_test: np.ndarray
        - n: int
    """
    test_images = X_test[n : n+5]

    generated_image = model.predict(test_images)
    plt.figure(figsize=(14, 4))

    for i in range(4):
        plt.subplot(1, 5, i + 1)
        
        plt.imshow(test_images[0, i].reshape((28, 28)), cmap='gray')
        plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.title('Generated Image')
    plt.imshow(generated_image[0], cmap='gray')
    plt.axis('off')

    plt.show()