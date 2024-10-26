'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-10-01 12:43:58
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-10-01 12:44:09
 # @ Proyect: Aitea Building Lab
 # @ Description:Run and execute pipelines
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

import pandas as pd
import dill
import logging
import datetime

from utils.data_utils import synchronization_and_optimization
import models_warehouse

date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
__version__ = f"0.0.1 {date}"


class PipeExecutor(object):
    def __init__(self, path: str):
        """Runs a pipeline, loaded with a pickle. Incorporates different utilities

        Args:
            path (str): Pipeline picke path 
        """
        self.pipe = self.load_pipe(path)
    
    def get_query(self) -> dict:
        """Get query
        Returns:
            dict: Dictionary with original search
        """
        return self.pipe.get("training_query")

    def run_transform(self, X: pd.DataFrame):
        """Run a transform

        Args:
            X (pd.DataFrame): Data to transform

        Returns:
            pd.Dataframe: Transformed data
        """
        pipe_core = self.pipe.get("pipe")
        return pipe_core.transform(X)

    def run_predict(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Run a predict

        Args:
            X (pd.DataFrame): Data to prediction
            y (pd.DataFrame, optional): Target. Defaults to None.

        Returns:
            pd.DataFrame: Prediction
        """
        pipe_core = self.pipe.get("pipe")
        return pipe_core.predict(X, y)

    def get_params(self, position: int = 0) -> dict:
        """Get a pipe element parameters

        Args:
            position (int, optional): Element position. Defaults to 0.

        Returns:
            dict: Parameters
        """
        pipe_core = self.pipe.get("pipe")
        params = None
        try:
            params = pipe_core[position].get_params()
        except KeyError:
            logging.error(f"[Lib error], The {position} is not a position in the pipeline")
        return params
    
    def set_params(self, params: dict, position: int = 0):
        """Sets the parameters of a pipe position

        Args:
            params (dict): Parameteres
            position (int, optional): Element position. Defaults to 0.
        """
        pipe_core = self.pipe.get("pipe")
        try:
            pipe_core[position].set_params(params)
        except KeyError:
            logging.error(f"[Lib error], The {position} is not a position in the pipeline")
    
    def synchronization_and_optimization(self, X: pd.DataFrame) -> pd.DataFrame:
        """Synchronize the data obtained in influx

        Args:
            X (pd.DataFrame): Dataframe

        Returns:
            pd.DataFrame: Pivoted and synchronized data 
        """
        return synchronization_and_optimization(X)
    
    def load_pipe(self, path: str):
        """Load pickle

        Args:
            path (str): Path to pickle

        Returns:
            _type_: Pipeline
        """
        pipeline = None
        with open(path, "rb") as f:
            pipeline = dill.load(f)
        return pipeline
    
    
    def help(self):
        """
         __init__(self, path: str):
        Runs a pipeline, loaded with a pickle. Incorporates different utilities

        Args:
            path (str): Pipeline picke path 

        get_query(self) -> dict:
        Get query
        Returns:
            dict: Dictionary with original search
        
        run_transform(self, X: pd.DataFrame):
        Run a transform
        Args:
            X (pd.DataFrame): Data to transform

        Returns:
            pd.Dataframe: Transformed data
        
        run_predict(self, X: pd.DataFrame, y: pd.DataFrame = None):
        Run a predict

        Args:
            X (pd.DataFrame): Data to prediction
            y (pd.DataFrame, optional): Target. Defaults to None.

        Returns:
            pd.DataFrame: Prediction
        
        get_params(self, position: int = 0) -> dict:
        Get a pipe element parameters

        Args:
            position (int, optional): Element position. Defaults to 0.

        Returns:
            dict: Parameters
        
        set_params(self, params: dict, position: int = 0):
        Sets the parameters of a pipe position

        Args:
            params (dict): Parameteres
            position (int, optional): Element position. Defaults to 0.
        
        
        synchronization_and_optimization(self, X: pd.DataFrame) -> pd.DataFrame:
        Synchronize the data obtained in influx

        Args:
            X (pd.DataFrame): Dataframe

        Returns:
            pd.DataFrame: Pivoted and synchronized data 
        
        load_pipe(self, path: str):
        Load pickle

        Args:
            path (str): Path to pickle

        Returns:
            _type_: Pipeline
        """


