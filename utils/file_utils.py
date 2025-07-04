'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-08-06 13:59:53
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-08-06 14:16:34
 # @ Proyect: Aitea Building Lab
 # @ Description:
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''



import json
from json.decoder import JSONDecodeError
from  loguru import logger

@logger.catch
def load_json_file(path: str) -> dict:
    """Load json file

    Args:
        path (str): File path

    Returns:
        dict: Json as a dictionary 
    """
    json_file = None
    try:
        f = open(path, mode="r", encoding="utf-8")
        json_file = json.loads(f.read())
        f.close()
        logger.debug(f"Read file: {path}")
    except Exception as e:
        logger.error(
            f"Error reading file {path}: {e}"
        )
    return json_file

