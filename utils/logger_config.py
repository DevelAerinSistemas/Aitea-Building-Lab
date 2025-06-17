'''
 # @ Project: AItea-Simulation
 # @ Author: Jose Luis Blanco
 # @ Create Time: 2024-12-23 18:51:38
 # @ Description: Logger script setup
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Jose Luis Blanco
 # @ Modified time: 2024-12-23 18:52:03
 # @ Copyright (c): 2024 - 2024 Aitea Tech S. L. copying, distribution or modification not authorised in writing is prohibited
 '''




from loguru import logger
import sys

def setup_logger(level: str = "DEBUG", logfile: str = None):
    logger.remove()
    logger.add(sys.stderr, level=level)
    if logfile:
        logger.add(logfile, level=level)

def get_logger():
    return logger