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
from loguru import logger
import pandas as pd
from datetime import datetime
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
        logger.error(
            f"Error: The file '{path}' is not well formed or has invalid JSON format.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    else:
        ok = check_configure(data)
        if not ok:
            data = None
            logger.error(
                "Error: Essential elements are missing in the pipeline configuration. Make sure it looks like this:  'pipe_name': { 'steeps': { 'analytic_file_name.Trasform_or_ModelCLass': {} }, 'datetime': { 'start': '', 'freq': ''}")
    return data


def check_configure(data_json: dict) -> bool:
    """Check if the configuration is correct

    Args:
        data_json (dict): Configuration plan dictionary

    Returns:
        bool: Correct or incorrect
    """
    ok = True
    for key, values in data_json.items():
        if isinstance(values, dict):
            keys = list(values.keys())
            if 'steeps' not in keys or not 'freq_info' in keys or not 'training_query' in keys:
                ok = False
    return ok


def lab_fit(data: pd.DataFrame, pipe_core: dict):
    logger.info(f"Starting pipe fitting for {pipe_core}")
    try:
        pipe = pipe_core["pipe"]
        query = pipe_core["training_query"]
        name = pipe_core["name"]
        try:
            logger.info("Start fit")
            pipe.fit(data)
        except Exception as err:
            logger.error(f" Fail fit {err}")
            return "InsufficientDataError"
    except Exception as err:
        return "KeyError"
    training_pipe = {"pipe_name": name, "pipe": pipe, "training_query": query,  "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    return training_pipe

def pipe_save(pipe_data:dict):
    dill.settings['recurse'] = True
    name = pipe_data.get("pipe_name")
    path = "training_models/" + name + ".pkl"
    with open(path, "wb") as f:
        dill.dump(pipe_data, f)

def load_pipe(path: str):
    pipeline = None
    with open(path, "rb") as f:
        pipeline = dill.load(f)
    return pipeline