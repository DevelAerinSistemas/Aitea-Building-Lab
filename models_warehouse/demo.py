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
import pandas as pd
from utils.logger_config import get_logger
logger = get_logger()

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
    
    def fit(self, X: Any, y: Any = None) -> None:
        pass
    
    def transform(self, X: Any) -> np.ndarray:
        pass
    
    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        pass

    def get_info(self) -> str:
        return f"{self.__class__.__name__} v.{self.LIBRARY_VERSION}"
    
    def get_all_attributes(self) -> dict:
        return self.parameters

    def get_results(self) -> dict:
        return self.results_dictionary

    @logger.catch
    def fuse_data_sources(self, connections: dict, training_info: dict) -> Any:
        """Example

        Args:
            connections (dict): Dictionary with connectors and connections
            training_info (dict): Dictionary with queries and datafiles

        Returns:
            Any: Example output of data fusing
        """
        dataframe_list = []
        for data_source, data_source_info in training_info.items():
            if data_source == "influxdb":
                logger.info(f"⚙️ Retrieving data from datasource '{data_source}'")
                dataframe_list.append(
                    connections[data_source]["connector"].query(
                        query=data_source_info, 
                        pandas=True, 
                        stream=False
                    )
                )
                logger.info(f"✅ Query data retrieval finished for datasource '{data_source}'")
            elif data_source == "postgresql":
                logger.info(f"⚙️ Retrieving data from datasource '{data_source}'")
                dataframe_list.append(
                    connections[data_source]["connector"].query_to_df(
                        query=data_source_info
                    )
                )
            elif data_source == "local":
                for datafile in data_source_info:
                    logger.info(f"⚙️ Retrieving data from local datafile at '{datafile}'")
                    source_type = datafile.split(".")[-1]
                    if source_type == "csv":
                        dataframe_list.append(pd.read_csv(datafile))
                    elif source_type == "json":
                        dataframe_list.append(pd.read_json(datafile))
                    elif source_type in ["xlsx", "ods"]:
                        dataframe_list.append(pd.read_excel(datafile))
                    else:
                        logger.warning(f"⚠️ Reading data from source of type '{source_type}' not implemented yet")                
            if len(dataframe_list) > 0:
                total_dataframe = pd.concat(dataframe_list, ignore_index=True)  # Added ignore_index=True for better concatenation
        return total_dataframe


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
        return f"{self.__class__.__name__} v.{self.LIBRARY_VERSION}"
    
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
        prediction_results = pd.DataFrame()
        self.results_dictionary = {"demomodel_analysis": prediction_results}
        return prediction_results

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        pass

    def predict_proba(self, X: Any) -> Any:
        pass

    def get_info(self) -> str:
        return f"{self.__class__.__name__} v.{self.LIBRARY_VERSION}"
    
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