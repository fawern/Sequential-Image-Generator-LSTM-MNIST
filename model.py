"""
Model module for the LSTM model
"""

import numpy as np
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Reshape, Input # type: ignore

class NextImagePredictor(Sequential):    
    """
    Model class for the LSTM model
    """
    def __init__(
            self, 
            X_train: np.ndarray, 
            y_train: np.ndarray, 
        ) -> None:
        """
        Initialize the model
        """
        super().__init__()  

        self.X_train = X_train
        self.y_train = y_train

        input_shape = X_train.shape[1:]
        
        self.add(Input(shape=input_shape))
        self.add(LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True))
        self.add(LSTM(64, activation='relu'))
        self.add(Dense((28*28), activation='linear'))
        self.add(Reshape((28, 28)))
    
    def train(
            self, 
            optimizer: str,
            iters: int
        ) -> None:
        """
        Train the model

        Args:
            - optimizer: str
            - iters: int
        """
        self.compile(
            optimizer=optimizer, loss='mean_squared_error', metrics=['mse']
        )
        self.fit(self.X_train, self.y_train, epochs=iters)
