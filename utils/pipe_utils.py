'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-09-26 12:25:32
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-09-26 12:26:22
 # @ Proyect: Aitea Building Lab
 # @ Description:
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

import json
import time
from loguru import logger
import pandas as pd
from datetime import datetime
import joblib
import dill



def read_json_schedule_plan(path: str) -> dict:
    """Read a json to have an execution plan, check if it has the basic configuration

    Returns:
        dict: Configuration plan execution
    """
    data = None
    try:
        with open(path, 'r') as archivo:
            data = json.load(archivo)
    except FileNotFoundError:
        logger.error(f"Error: The '{path}' not exist.")
    except json.JSONDecodeError:
        logger.error(f"Error: The file '{path}' is not well formed or has invalid JSON format.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    else:
        ok = check_configure(data)
        if not ok:
            data = None
            logger.error("Error: Essential elements are missing in the pipeline configuration. Make sure it looks like this:  'pipe_name': { 'steps': { 'analytic_file_name.Trasform_or_ModelCLass': {} }, 'datetime': { 'start': '', 'freq': ''}")
    return data


def check_configure(data_json: dict, json_keys: list[str] = ["steps","data_sources"]) -> bool:
    """Check if the configuration is correct

    Args:
        data_json (dict): Configuration plan dictionary
        json_keys (list[str]): List of keys a JSON pipeline must have

    Returns:
        bool: Correct or incorrect
    """
    ok = True
    for key, values in data_json.items():
        if isinstance(values, dict):
            keys = list(values.keys())
            if any(jk not in keys for jk in json_keys):
                ok = False
    return ok

def lab_fit(data: pd.DataFrame, pipe_core: dict, fit_and_predict: bool = False):
    """Make a pipe fit

    Args:
        data (pd.DataFrame): Data
        pipe_core (dict): Pipe core 
        fit_and_predict (bool): Flag to indicate if fit and predict should be performed

    Returns:
        dictionary: A fit pipe in a dictionary with the following keys 
        - pipe_name: Name of the pipe
        - pipe: The fitted pipe object
        - training_query: The query used for training
        - date: The date and time of fitting
    """
    timi_i = time.time()
    logger.info(f"⚙️ Starting fitting process for pipe_core:\n{pipe_core}")
    try:
        pipe = pipe_core["pipe"]
        query = pipe_core["training_query"]
        name = pipe_core["name"]
        try:
            logger.info("⚙️ Starting fitting process")
            if fit_and_predict:
                logger.info("⚙️ Starting fitting and predicting process")
                pipe.fit_predict(data)
            else:
                logger.info("⚙️ Starting fitting")
                pipe.fit(data)
        except Exception as err:
            logger.error(f"❌ Fittin process failed: {err}")
            return "InsufficientDataError"
    except Exception as err:
        return "KeyError"
    training_pipe = {
        "pipe_name": name, 
        "pipe": pipe, 
        "training_query": query,  
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    logger.info(f"✅ Pipe fitting process finished successfully for task '{pipe_core}' in {time.time() - timi_i}")
    return training_pipe

def pipe_save(pipe_data:dict, save_in_joblib: bool = False) -> str:
    """Save a pipe

    Args:
        pipe_data (dict): Pipe dict, info + pipe core
        save_in_joblib (bool, optional): Flag to indicate if the pipe should be saved in joblib format. Defaults to False.
    Returns:
        str: Path to the saved pipe    
    """
    if save_in_joblib:
        path = "lib/" + pipe_data.get("pipe_name") + ".joblib"
        joblib.dump(pipe_data, path)
    else:
        dill.settings['recurse'] = True
        name = pipe_data.get("pipe_name")
        path = "lib/" + name + ".pkl"
        with open(path, "wb") as f:
            dill.dump(pipe_data, f)
    return path

def load_pipe(path: str):
    """Load a pipe 

    Args:
        path (str): Pipe path file

    Returns:
        _type_: Pipe
    """
    pipeline = None
    with open(path, "rb") as f:
        pipeline = dill.load(f)
    return pipeline