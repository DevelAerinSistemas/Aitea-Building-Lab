'''
 # @ Author: Héctor Bermúdez Castro <hector.bermudez@aitea.tech>
 # @ Create Time: 2025-06-18
 # @ Modified by: Héctor Bermúdez Castro
 # @ Modified time: 2025-06-18
 # @ Project: Aitea Building Lab
 # @ Description: Analytic template
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from typing import Any
import numpy as np
from loguru import logger

from metaclass.templates import MetaFuse, MetaTransform, MetaModel
import datetime

class DemoFuse(MetaFuse):

    def __init__(self, **parameters):
        """Example

        Args:
            parameters (kwargs): Example parameters for the template.
        """
        super().__init__()
        self.LIBRARY_VERSION = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.LIBRARY_FIT_DATE = self.LIBRARY_VERSION
        self.parameters = parameters
        self.results_dictionary = {}
    
    # ABSTRACT METHODS
    
    @logger.catch
    def fuse_data_sources(self) -> Any:
        """Example

        Returns:
            Any: Example output of data fusing
        """
        Xs = []
        for source, data in self.data_sources.items():
            try:
                if source == "local":
                    for local_source in data:
                        source_type = local_source.split(".")[-1]
                        if source_type == "csv":
                            Xs.append(pd.read_csv(local_source))
                        elif source_type == "json":
                            Xs.append(pd.read_json(local_source))
                        elif source_type in ["xlsx", "ods"]:
                            Xs.append(pd.read_excel(local_source))
                        else:
                            logger.warning(f"Reading data from source of type '{source_type}' not implemented yet")
                else:
                    Xs.append(pd.DataFrame(data))
            except Exception as err:
                logger.warningf(f"Error converting data from source '{source}' into pd.DataFrame: {error}")
        return pd.concat(Xs, ignore_index=True)

class DemoTransform(MetaTransform):

    def __init__(self, **parameters):
        """Example

        Args:
            parameters (kwargs): Example parameters for the template.
        """
        super().__init__()
        self.LIBRARY_VERSION = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.LIBRARY_FIT_DATE = self.LIBRARY_VERSION
        self.parameters = parameters
        self.results_dictionary = {}
    
    # ABSTRACT METHODS

    def fit(self, X: Any, y: Any = None) -> None:
        pass
    
    def transform(self, X: Any) -> np.ndarray:
        pass
    
    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        pass

    def get_info(self) -> str:
        return f"{self.__class__.__name__} v.{LIBRARY_VERSION}"
    
    def get_all_attributes(self) -> dict:
        return self.parameters

    def get_results(self) -> dict:
        return self.results_dictionary
    
    # NON ABSTRACT METHODS

    @logger.catch()    
    def update_parameters(self, **attributes):
        """
        Update the model's attributes with the given values.

        Args:
            **attributes: A dictionary of attributes to update.
        """
        for name, value in attributes.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                logger.warning(f"Attribute '{name}' not found in {self.__class__.__name__}.")

    @logger.catch()
    def generate_query_params(self):
        pass

class DemoModel(MetaModel):

    def __init__(self, **parameters):
        """Example

        Args:
            parameters (kwargs): Example parameters for the template.
        """
        super().__init__()
        self.LIBRARY_VERSION = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.LIBRARY_FIT_DATE = self.LIBRARY_VERSION
        self.parameters = parameters
        self.results_dictionary = {}
    
    # ABSTRACT METHODS

    def fit(self, X: Any, y: Any = None) -> None:
        pass
    
    def predict(self, X: Any) -> np.ndarray:
        pass

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        pass

    def predict_proba(self, X: Any) -> Any:
        pass

    def get_info(self) -> str:
        return f"{self.__class__.__name__} v.{LIBRARY_VERSION}"
    
    def get_all_attributes(self) -> dict:
        return self.parameters

    def get_results(self) -> dict:
        return self.results_dictionary
    
    # NON ABSTRACT METHODS

    @logger.catch()    
    def update_parameters(self, **attributes):
        """
        Update the model's attributes with the given values.

        Args:
            **attributes: A dictionary of attributes to update.
        """
        for name, value in attributes.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                logger.warning(f"Attribute '{name}' not found in {self.__class__.__name__}.")
    
    @logger.catch()
    def generate_query_params(self):
        pass

if __name__ == "__main__":

    examples = [
        DummyTransform(example_parameter=1.0),
        DummyModel(example_parameter=1.0)
    ]
    for e in examples:
        logger.info(e.get_info())
        logger.info(e.get_all_attributes())
        logger.info(e.get_results())