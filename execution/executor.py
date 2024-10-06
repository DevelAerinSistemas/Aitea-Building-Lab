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
        self.pipe = self.load_pipe(path)
    
    def get_query(self) -> dict:
        return self.pipe.get("training_query")

    def run_transform(self, X: pd.DataFrame):
        pipe_core = self.pipe.get("pipe")
        return pipe_core.transform(X)

    def run_predict(self, X: pd.DataFrame, y: pd.DataFrame = None):
        pipe_core = self.pipe.get("pipe")
        return pipe_core.predict(X, y)

    def get_params(self, position: int = 0) -> dict:
        pipe_core = self.pipe.get("pipe")
        params = None
        try:
            params = pipe_core[position].get_params()
        except KeyError:
            logging.error(f"[Lib error], The {position} is not a position in the pipeline")
        return params
    
    def set_params(self, params: dict, position: int = 0):
        pipe_core = self.pipe.get("pipe")
        try:
            pipe_core[position].set_params(params)
        except KeyError:
            logging.error(f"[Lib error], The {position} is not a position in the pipeline")
    
    def synchronization_and_optimization(self, X: pd.DataFrame) -> pd.DataFrame:
        return synchronization_and_optimization(X)
    
    def load_pipe(self, path: str):
        pipeline = None
        with open(path, "rb") as f:
            pipeline = dill.load(f)
        return pipeline
    


