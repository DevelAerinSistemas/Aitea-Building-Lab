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



def get_configuration(global_configuration: str = 'config/global_config.json', section: str = None) -> dict:
    """Get configuration from file

    Args:
        global_configuration (str, optional): Configuration file path. Defaults to 'config/global_config.json'.
        section (str, optional): Configuration section. Defaults None. 
    Returns:
        dict: Configuration dict
    """
    config_dictionary = None
    try:
        with open(global_configuration) as config_file:
            config_dictionary = json.load(config_file)
            if section:
                config_dictionary = config_dictionary.get(section, {})
    except FileNotFoundError:
        logger.error(f"Configuration file not found : {global_configuration}") 
    except JSONDecodeError:
        logger.error(f" Wrong or poorly formatted json : {global_configuration}")
    return config_dictionary