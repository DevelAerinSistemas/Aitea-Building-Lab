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
from utils.file_utils import load_json_file
from utils.so_utils import create_so
from exceptions.fit_exception import InsufficientDataError


class PipelineManager(object):
    def __init__(self, configuration_definition_file: str):
        """Create and manage scikit-learn pipelines

        Args:
            configuration_definition_file (str): Configuration file 
        """
        self.configuration = read_json_schedule_plan(
            configuration_definition_file)
        if self.configuration is None:
            logger.error(
                f'"Error in configuration, the pipes cannot be trained')
        self.pipes = {}

    def create_pipelines(self, influx_connection: InfluxDBConnector, bucket_not_considered: set):
        """Create all pipes
        """
        if self.configuration:
            for pipe_name, pipe_values in self.configuration.items():
                pipe = self.create_one_pipeline(pipe_values)
                query = pipe_values.get("training_query")
                bucket = query.get("bucket", {}).get("bucket")
                final_buckets = self.get_valid_buckets(bucket, influx_connection, bucket_not_considered)
                freq_info = pipe_values.get("freq_info")
                query["bucket"]["bucket"] = final_buckets
                self.pipes[pipe_name] = {
                    "pipe":  pipe, "training_query": query, "freq_info": freq_info}

    def create_one_pipeline(self, pipeline_details: dict) -> Pipeline:
        """Create one pipe

        Args:
            pipeline_details (dict): Pipeline configuration details 

        Returns:
            Pipeline: One scikit-learn pipeline
        """
        pipe_parts = []
        steeps = pipeline_details.get("steeps")
        for element, params in steeps.items():
            one_instance = self._generate_instance(element, params)
            if one_instance is None:
                logger.critical(f"Error creating instance for {element}.")
                exit(1)
            pipe_parts.append((element, one_instance))
        return Pipeline(pipe_parts)
    
    def get_valid_buckets(self, actual_buckets: list, influx_connection: InfluxDBConnector, bucket_not_considered: set) -> list:
       """Get all buckets from the configuration.
       Args:
           actual_buckets (list): List of actual buckets.
           influx_connection (InfluxDBConnector): InfluxDB connection object.
       Returns:
           list: List of all buckets.
       """
       all_buckets = []
       if "all_buckets" in actual_buckets or "all" in actual_buckets:
           all_buckets = set(influx_connection.get_bucket_list()) - bucket_not_considered   
           all_buckets = [bucket for bucket in all_buckets if not bucket.startswith("_")]
       else:   
           all_buckets = actual_buckets
       return list(all_buckets)

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
        data_conection = load_json_file(os.getenv("GLOBAL_DATA"))
        self.generate_so = generate_so
        self.save_in_joblib = save_in_joblib
        temporan_influx = InfluxDBConnector(data_conection)
        temporan_influx.load_configuration()
        _, _ = temporan_influx.connect(True)
        bucket_not_considered = set(data_conection.get("bucket_not_considered", []))
        self.create_pipelines(temporan_influx, bucket_not_considered)
        for name, pipes_data in self.pipes.items():
            influx = InfluxDBConnector(data_conection)
            influx.load_configuration()
            _, _ = influx.connect(True)
            pipes_data["connection"] = influx
         
        

    def data_preparation(self, pipe_data: dict) -> pd.DataFrame:
        """Prepare data for the pipeline execution.

        Args:
            pipe_data (dict): Pipeline data containing the query

        Returns:
            pd.DataFrame: Data to fit the pipe
        """
        query_buckets = copy.deepcopy(pipe_data.get("training_query"))
        buckets = query_buckets.get("bucket", {}).get("bucket")
        influx_connection = pipe_data.get("connection")
        dataframe_list = list()
        total_dataframe = None
        if isinstance(buckets, list):
            for bucket in buckets:
                query_buckets["bucket"]["bucket"] = bucket
                logger.info(f" Start query in  {bucket}")
                query = influx_connection.compose_influx_query_from_dict(query_buckets)
                stream_data = influx_connection.query(
                    query=query, pandas=True, stream=False)
                if stream_data is None:
                    continue
                logger.info(f" End query in {bucket}")
                stream_data["bucket"] = bucket
                dataframe_list.append(stream_data)
            if len(dataframe_list) > 0:
                total_dataframe = pd.concat(dataframe_list, ignore_index=True)  # Added ignore_index=True for better concatenation
        return total_dataframe

    def pipes_executor(self, testing: bool = False):
        """Executes the pipeline tasks using multiprocessing.

        Args:
            testing (bool, optional): Flag to indicate if the execution is for testing purposes (fit and predict). Defaults to False.
        """
        with multiprocessing.Pool(processes=self.total_processing) as pool:
            for pipe_name, pipe_info in self.pipes.items():
                logger.info(
                    f"Acquiring influx data to perform task to {pipe_name}")
                data = self.data_preparation(pipe_info)
                if data is None:
                    logger.warning("Empty data, can't do training")
                    continue
                else:
                    pipe_core = {"name": pipe_name, "pipe": pipe_info.get(
                        "pipe"), "training_query":  pipe_info.get("training_query")}
                    pool.apply_async(lab_fit, args=(data, pipe_core, testing),
                                     callback=self.task_handler,
                                     error_callback=self.error_handler)
                    logger.info(f"Geting fit to {pipe_name}")
            pool.close()
            pool.join()

    def task_handler(self, result):
        """Handles the result of the pipeline fitting process.

        Args:
            result: The result of the fitting process.
        """
        if result == "InsufficientDataError":
            logger.critical(f" Not enough data to train")
        elif result == "KeyError":
            logger.critical(f" The pipe is malformed, keys are missing")
        else:
            training_file = pipe_save(result, self.save_in_joblib)
            logger.info(f" End pipe fit. It is saved {training_file}")
            if self.generate_so:
                logger.info("Creating shared object")
                self.launch_create_so(training_file)

    def error_handler(self, result):
        """Handles errors that occur during the fitting process.

        Args:
            result: The error result from the fitting process.
        """
        logger.error(f"Error in fit {result}")

    def launch_create_so(self, model_path: str) -> None:
        """Launches the creation of a shared object.

        Args:
            model_path (str): Path to the model file.
        """
        create_so(model_path=model_path)
        logger.info(f"Shared object created at {model_path}")
    
    
   
            
        
        

if __name__ == "__main__":
    pipe = PipelineExecutor("pipes_schedules/pipe_plan.json", generate_so=True, save_in_joblib=False)
    pipe.pipes_executor(testing=False)
    logger.info("Pipeline execution completed.")