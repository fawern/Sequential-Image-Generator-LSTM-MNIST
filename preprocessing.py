"""
Data preprocessing module for MNIST
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import Tuple

class SequentialImageDataGenerator:
    """
    Data generator for sequence data
    """
    def __init__(self, img_df: pd.DataFrame) -> None:
        self.new_df = pd.DataFrame(columns=['input1', 'input2', 'input3', 'input4', 'output'])
        self.img_df = img_df

    @staticmethod
    def filter_by_label(df: pd.DataFrame, label: int) -> pd.Series:
        """
        Filter the dataframe by label

        Args: 
            - df: pandas dataframe
            - label: int

        Returns:
            - series of images with the given label
        """
        return df[df['labels'] == label]['imgs']

    def generate_data_for_labels(
            self, 
            df: pd.DataFrame, 
            label_numbers: list[int]
        ) -> pd.DataFrame:
        """
        Generate data for the given labels

        Args:
            - df: pandas dataframe
            - label_numbers: list of labels

        Returns:
            - pandas dataframe with the generated data
        """
        min_label_count = min(df[df['labels'].isin(label_numbers)]['labels'].value_counts())
        for i in range(min_label_count):
            self.new_df.loc[len(self.new_df)] = [self.filter_by_label(df, label_numbers[j]).iloc[i] for j in range(len(label_numbers))]
        return self.new_df

    @staticmethod
    def generate_formula_data(n): 
        """
        Generate data for the given formula

        Args:
            - n: int

        Returns:
            - tuple of data
        """
        return n, (n + 1), (n + 2), (n + 3), (n + 4)        

    def generate_data(self) -> pd.DataFrame:
        """
        Generate data for the given labels

        Returns:
            - pandas dataframe with the generated data
        """
        data_formats = [list(self.generate_formula_data(i)) for i in range(6)]
        for data_format in data_formats:
            self.generate_data_for_labels(self.img_df, data_format)

        return self.new_df

    def re_scale_data(self) -> Tuple[np.ndarray, np.ndarray]: 
        """
        Re-scale the data

        Returns:
            - numpy array with the re-scaled data
        """
        generated_df = self.generate_data() 

        X = np.stack([np.array(generated_df[col].tolist()) for col in ['input1', 'input2', 'input3', 'input4']], axis=-1)
        y = np.stack([np.array(generated_df[col].tolist()) for col in ['output']], axis=-1)

        X = X.reshape(-1, 28, 28, 4)
        y = y.reshape(-1, 28, 28, 1)

        X = np.transpose(X, (0, 3, 1, 2))
        y = np.transpose(y, (0, 3, 1, 2))

        X = X.reshape((X.shape[0]), 4, 28*28)

        y = y.reshape((y.shape[0]), 28, 28)

        return X, y

    def train_test_split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        """
        Split the data into train and test sets

        Returns:
            - numpy array with the train and test data
        """
        X, y = self.re_scale_data() 
        return train_test_split(X, y, test_size=0.2, random_state=23)
