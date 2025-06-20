'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-09-26 12:39:46
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-09-26 12:39:50
 # @ Proyect: Aitea Building Lab
 # @ Description: Pipeline creation and execution main class
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from sklearn.pipeline import Pipeline
import multiprocessing

import copy
from typing import Any
import importlib
from loguru import logger
import pandas as pd
import subprocess
from dotenv import load_dotenv
import os

from database_tools.influxdb_connector import InfluxDBConnector
from utils.pipe_utils import read_json_schedule_plan, lab_fit, pipe_save
from utils.file_utils import load_json_file, get_configuration
from utils.logger_config import get_logger
from utils.so_utils import create_so
from exceptions.fit_exception import InsufficientDataError


logger = get_logger()

class PipelineManager(object):

    @logger.catch
    def __init__(self, configuration_definition_file: str):
        """Create and manage scikit-learn pipelines

        Args:
            configuration_definition_file (str): Configuration file 
        """
        self.configuration = read_json_schedule_plan(configuration_definition_file)
        if self.configuration is None:
            logger.error(f'"Error in configuration, the pipes cannot be trained')
        self.pipes = {}

    @logger.catch
    def create_pipelines(self, influxdb_conn: InfluxDBConnector, buckets_not_considered: set):
        """Create all pipes
        """
        if self.configuration:
            for pipe_name, pipe_values in self.configuration.items():
                pipe = self.create_one_pipeline(pipe_values)
                query = pipe_values.get("training_query")
                buckets = query.get("buckets")
                final_buckets = self.get_valid_buckets(
                    buckets, 
                    influxdb_conn, 
                    buckets_not_considered
                )
                query["buckets"] = final_buckets
                freq_info = pipe_values.get("freq_info")

                self.pipes[pipe_name] = {
                    "pipe":  pipe, 
                    "training_query": query, 
                    "freq_info": freq_info,
                }
                if "query_parts" in query:
                    self.pipes[pipe_name]["query_params"] = pipe.steps[0].generate_query_params()

    @logger.catch
    def create_one_pipeline(self, pipeline_details: dict) -> Pipeline:
        """Create one pipe

        Args:
            pipeline_details (dict): Pipeline configuration details 

        Returns:
            Pipeline: One scikit-learn pipeline
        """
        pipe_parts = []
        steps = pipeline_details.get("steps")
        for element, params in steps.items():
            one_instance = self._generate_instance(element, params)
            if one_instance is None:
                logger.critical(f"Error creating instance for {element}.")
                exit(1)
            pipe_parts.append((element, one_instance))
        return Pipeline(pipe_parts)
    
    @logger.catch
    def get_valid_buckets(self, actual_buckets: list, influxdb_conn: InfluxDBConnector, buckets_not_considered: set) -> list:
       """Get all buckets from the configuration.
       Args:
           actual_buckets (list): List of actual buckets.
           influxdb_conn (InfluxDBConnector): InfluxDB connection object.
       Returns:
           list: List of all buckets.
       """
       all_buckets = []
       if "all_buckets" in actual_buckets or "all" in actual_buckets:
           all_buckets = set(influxdb_conn.get_bucket_list()) - buckets_not_considered   
           all_buckets = [bucket for bucket in all_buckets if not bucket.startswith("_")]
       else:   
           all_buckets = actual_buckets
       return list(all_buckets)

    @logger.catch
    def _generate_instance(self, class_path: str, class_attributes: dict) -> Any:
        """Create an instance

        Args:
            class_path (str): Class path
            class_attributes (dict): Class attributes

        Returns:
            _type_: Instance
        """
        instance = None
        try:
            class_elements = class_path.split(".")
            module_name = "models_warehouse." + class_elements[0]
            module = importlib.import_module(module_name)
            the_class = getattr(module, class_elements[1])
            instance = the_class(**class_attributes)
        except ModuleNotFoundError as e: 
            logger.error(f"Error: The module'{class_elements[0]}', not found or other error in module: {e}")
        except AttributeError as e:
            logger.error(f"Error: The class '{class_elements[1]}' not found. {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        return instance


class PipelineExecutor(PipelineManager):

    @logger.catch
    def __init__(self, configuration_definition_file: str, total_processing: int = 4, generate_so: bool = True, save_in_joblib: bool = False):
        """Initializes the PipelineExecutor with configuration and processing options.

        Args:
            configuration_definition_file (str): Configuration file path.
            total_processing (int, optional): Number of processes to use. Defaults to 4.
            generate_so (bool, optional): Flag to generate shared object. Defaults to True.
            save_in_joblib (bool, optional): Flag to save in joblib format. Defaults to False.
        """
        super().__init__(configuration_definition_file)
        self.total_processing = total_processing
        self.generate_so = generate_so
        self.save_in_joblib = save_in_joblib
        config = get_configuration()
        influxdb_conn = InfluxDBConnector()
        buckets_not_considered = set(config.get("buckets_not_considered", []))
        self.create_pipelines(influxdb_conn, buckets_not_considered)
        for name, pipes_data in self.pipes.items():
            pipes_data["connection"] = influxdb_conn
        
    @logger.catch
    def data_preparation(self, pipe_data: dict) -> pd.DataFrame:
        """Prepare data for the pipeline execution.

        Args:
            pipe_data (dict): Pipeline data containing the query

        Returns:
            pd.DataFrame: Data to fit the pipe
        """
        training_query = copy.deepcopy(pipe_data.get("training_query"))
        buckets = training_query.get("buckets")
        influxdb_conn = pipe_data.get("connection")
        query_params = pipe_data.get("query_params")
        dataframe_list = list()
        total_dataframe = None
        if isinstance(buckets, list):
            for bucket in buckets:
                logger.info(f"Starting query generation for bucket '{bucket}'")
                query_dict = {"bucket":{"bucket":bucket}}
                for k,v in training_query.items():
                    if k!="buckets":
                        query_dict[k] = v
                query = influxdb_conn.compose_influx_query_from_dict(query_dict)
                query_parts = training_query.get("query_parts",[])
                if query_parts:
                    query_parts.prepend(query)
                    query = "\n  |>".join(query_parts).format(**query_params)
                logger.info(f"Retrieving data from InfluxDB using query:\n{query}")
                stream_data = influxdb_conn.query(
                    query=query, 
                    pandas=True, 
                    stream=False
                )
                if stream_data is None:
                    continue
                logger.info(f" End query in {bucket}")
                stream_data["bucket"] = bucket
                dataframe_list.append(stream_data)
            if len(dataframe_list) > 0:
                total_dataframe = pd.concat(dataframe_list, ignore_index=True)  # Added ignore_index=True for better concatenation
        return total_dataframe

    @logger.catch
    def pipes_executor(self, testing: bool = False):
        """Executes the pipeline tasks using multiprocessing.

        Args:
            testing (bool, optional): Flag to indicate if the execution is for testing purposes (fit and predict). Defaults to False.
        """
        with multiprocessing.Pool(processes=self.total_processing) as pool:
            for pipe_name, pipe_info in self.pipes.items():
                logger.info(f"Acquiring influx data to perform task '{pipe_name}'")
                data = self.data_preparation(pipe_info)
                if data is None:
                    logger.warning("Empty data, can't do training")
                    continue
                else:
                    pipe_core = {
                        "name": pipe_name, 
                        "pipe": pipe_info.get("pipe"), 
                        "training_query": pipe_info.get("training_query")
                    }
                    pool.apply_async(
                        lab_fit, 
                        args=(data, pipe_core, testing),
                        callback=self.task_handler,
                        error_callback=self.error_handler
                        )
                    logger.info(f"Geting fit to {pipe_name}")
            pool.close()
            pool.join()

    @logger.catch
    def task_handler(self, result):
        """Handles the result of the pipeline fitting process.

        Args:
            result: The result of the fitting process.
        """
        if result == "InsufficientDataError":
            logger.critical(f" Not enough data to train")
        elif result == "KeyError":
            logger.critical(f"The pipe is malformed, keys are missing")
        else:
            training_file = pipe_save(result, self.save_in_joblib)
            logger.info(f"End pipe fit. It is saved {training_file}")
            if self.generate_so:
                logger.info("Creating shared object")
                self.launch_create_so(training_file)

    @logger.catch
    def error_handler(self, result):
        """Handles errors that occur during the fitting process.

        Args:
            result: The error result from the fitting process.
        """
        logger.error(f"Error in fit {result}")

    @logger.catch
    def launch_create_so(self, model_path: str) -> None:
        """Launches the creation of a shared object.

        Args:
            model_path (str): Path to the model file.
        """
        create_so(model_path=model_path)
        logger.info(f"Shared object created at {model_path}")
