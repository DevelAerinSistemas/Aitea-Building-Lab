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
from loguru import logger

from sklearn.base import BaseEstimator, TransformerMixin


class MetaFuse(ABC, BaseEstimator, TransformerMixin):
    """Metaclass to be used in data fusing
    """
    def __init__(self, data_sources: dict = {}) -> None:
        """Constructor of the class

        Args:
            data_sources (dict): _description
        """
        self.data_sources = data_sources
    
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
        """Transform the data

        Args:
            X (Any): Pandas o numpy with data to transform
        
        Returns:
            np.ndarray: Transformed data
        """
        pass
    
    @abstractmethod
    def fit_transform(self, X: Any, y: Any) -> Any:
        """First training is done and then a transformation

        Args:
            X (Any):Pandas o numpy with data to predict
            y (Any): Pandas o numpy with training data

        Returns:
            np.ndarray: Predict
        """
        pass

    @abstractmethod
    def fuse_data_sources(self) -> Any:
        """_summary

        Returns:    
            Any: np.ndarry or pd.DataFrame with data for fit
        """
        pass
    
    @logger.catch
    def get_data_from_source(self, data_source_name: str) -> Any:
        """Gets data from 'self.data_sources' with name 'data_source_name'

        Args:
            data_source_name (str): data source name, e.g. 'influxdb', 'postgresql', 'local', etc.

        Returns:    
            Any: np.ndarry or pd.DataFrame with data from data source 'data_source_name'
        """
        if data_source_name not in self.data_sources:
            logger.warning(f"Data source '{data_source_name}' not available. Available data_sources are: {list(self.data_sources.keys())}")
        return self.data_sources.get(data_source_name)


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
        """Transform the data

        Args:
            X (Any): Pandas o numpy with data to transform
        
        Returns:
            np.ndarray: Transformed data
        """
        pass
    
    @abstractmethod
    def fit_transform(self, X: Any, y: Any) -> Any:
        """First training is done and then a transformation

        Args:
            X (Any):Pandas o numpy with data to predict
            y (Any): Pandas o numpy with training data

        Returns:
            np.ndarray: Predict
        """
        pass

    @abstractmethod
    def get_info(self) -> str:
        """Get the information of the transformation and version

        Returns:
            str: Information about the transformation and version
        """
        pass
    
    @abstractmethod
    def get_all_attributes(self):
        """Returns all attributes of the model.
        
        Returns:
            dict: A dictionary containing all attributes of the model.
        """
        pass

    @abstractmethod
    def get_results(self) -> Any:
        """Get the results of the model

        Returns:
            Any: Results
        """
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
            y (Any): Pandas o numpy with training data

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
    
    def clean_model(self, model: Any):
        """Cleans the results of the model

        Returns:
            Any: Results
        """
        try:
            model._fit_X = None
        except Exception as err:
            logger.error(f"Error cleaning model {err}. The model most likely cannot be cleaned.")
        else:
            logger.info("Model cleaned")
    
    
    @abstractmethod
    def get_info(self) -> str:
        """Get the information of the transformation and version

        Returns:
            str: Information about the transformation and version
        """
        pass
    
    @abstractmethod
    def get_all_attributes(self):
        """Returns all attributes of the model.
        
        Returns:
            dict: A dictionary containing all attributes of the model.
        """
        pass

    @logger.catch()    
    def update(self, **attributes):
        """
        Update the model's attributes with the given values.

        Args:
            **attributes: A dictionary of attributes to update.
        """
        for name, value in attributes.items():
            if hasattr(self, name):
                setattr(self, name, value)
                logger.info(f"Attribute '{name}' updated in {self.__class__.__name__}.")
            else:
                logger.warning(f"Attribute '{name}' not found in {self.__class__.__name__}.")