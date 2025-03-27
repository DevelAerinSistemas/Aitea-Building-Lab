'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-09-09 11:15:44
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-09-09 11:15:48
 # @ Proyect: Aitea Building Lab
 # @ Description: Metaclasses and templates. 
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from typing import Any
import numpy as np
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class MetaTransform(ABC, BaseEstimator, TransformerMixin):
    """Metaclass to be used in transformations
    """
    
    @abstractmethod
    def fit(self, X: Any, y: Any= None) -> None:
        """_summary_

        Args:
            X (Any): _description_
            y (Any, optional): _description_. Defaults to None.
        """
        pass

    @abstractmethod
    def transform(self, X: Any) -> Any:
        pass



class MetaModel(ABC, BaseEstimator, TransformerMixin):
    """Metaclass to be used in models
    """
    
    @abstractmethod
    def fit(self, X: Any, y: Any= None) -> Any:
        """Perform model training

        Args:
            X (Any): Pandas o numpy  with training data
            y (Any, optional): Nothing just for compatibility . Defaults to None.
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict 

        Args:
            X (Any): Pandas o numpy with data to predict

        Returns:
            np.ndarray: Prediction
        """
        pass

    @abstractmethod
    def fit_predict(self, X: Any, y: Any) -> Any:
        """First training is done and then a prediction

        Args:
            X (Any):Pandas o numpy with data to predict

        Returns:
            np.ndarray: Predict
        """
        pass


    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """In the case of a classifier, the probability of belonging to each class or in the case of being able to give a probability of the prediction

        Args:
            X (Any): Pandas o numpy with data to predict

        Returns:
            np.ndarray: Probability Vector
        """
        pass
    
    @abstractmethod
    def get_results(self) -> Any:
        """Get the results of the model

        Returns:
            Any: Results
        """
        pass

