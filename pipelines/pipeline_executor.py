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
import pandas as pd
import subprocess

from dotenv import load_dotenv
load_dotenv()
import os

from utils.logger_config import get_logger
logger = get_logger()
from utils.pipe_utils import read_json_schedule_plan, pipe_save, lab_fit
from utils.file_utils import load_json_file
from utils.so_utils import create_so
from exceptions.fit_exception import InsufficientDataError

try:
    from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
    from aitea_connectors.connectors.postgresql_connector import PostgreSQLConnector
    AITEA_CONNECTORS = True
except ImportError:
    logger.warning(f"⚠️ Aitea Connectors are not available. Uncomplete functionality: only local files as a valid data source.")
    AITEA_CONNECTORS = False

class PipelineManager(object):

    @logger.catch
    def __init__(self, pipeline_config_file: str):
        """Create and manage scikit-learn pipelines

        Args:
            pipeline_config_file (str): Configuration file 
        """
        global_config_path = os.getenv("CONFIG_PATH")

        self.global_config = load_json_file(global_config_path)
        if self.global_config is None:
            logger.error(f"❌ Error loading global configuration, {self.__class__.__name__} could not be instantiated successfully")
        else:
            self.connections = self.create_connections()

        self.pipeline_config = read_json_schedule_plan(pipeline_config_file)
        logger.info(f"📢 Valid pipeline schedule configuration loaded successfully from '{pipeline_config_file}'")
        if self.pipeline_config is None:
            logger.error(f"❌ Error loading pipeline configuration, {self.__class__.__name__} could not be instantiated successfully")
        else:
            self.pipelines = self.create_pipelines()

    @logger.catch
    def create_connections(self) -> dict:
        """Create all connections
        Returns:
            dict: connections
        """
        connections = {}
        for conn_name in self.global_config.get("data_sources"):
            if conn_name == "local":
                connections[conn_name] = []
                for folder in self.global_config.get(conn_name):
                    if os.path.exists(folder):
                        connections[conn_name].append(folder)
                    else:
                        logger.warning(f"⚠️ Folder '{folder}' for training files does not exixt")
            elif AITEA_CONNECTORS:
                if conn_name == "influxdb":
                    connections[conn_name] = {"connector": InfluxDBConnector()}
                    connections.update(zip(("connection_status","connection_client"),connections[conn_name]["connector"].connect()))
                elif conn_name == "postgresql":
                    connections[conn_name] = {"connector": PostgreSQLConnector()}
                    connections.update(zip(("connection_status","connection_client"),connections[conn_name]["connector"].connect()))
                else:
                    logger.warning(f"⚠️ Datasource of type '{conn_name}' is not implemented yet")
            else:
                logger.warning(f"⚠️ Datasource of type '{conn_name}' is not implemented yet (either locally or using AITEA_CONNECTORS)")
        return connections

    @logger.catch
    def get_valid_buckets(self, candidate_buckets: list) -> list:
       """Get all buckets from the configuration.
       Args:
           candidate_buckets (list): List of actual buckets.
       Returns:
           list: List of all buckets.
       """
       valid_buckets = []
       if "all_buckets" in candidate_buckets or "all" in candidate_buckets:
           valid_buckets = set(self.connections["influxdb"]["connector"].get_bucket_list()) - self.global_config.get("influxdb",{}).get("buckets_not_considered",set())
           valid_buckets = [bucket for bucket in valid_buckets if not bucket.startswith("_")]
       else:
           valid_buckets = candidate_buckets
       return list(valid_buckets)

    @logger.catch
    def create_one_pipeline(self, pipeline_steps: dict) -> Pipeline:
        """Create one pipe

        Args:
            pipeline_steps (dict): Pipeline steps configuration details 

        Returns:
            Pipeline: One scikit-learn pipeline
        """
        pipe_parts = []
        for element, params in pipeline_steps.items():
            one_instance = self._generate_instance(element, params)
            if one_instance is None:
                logger.critical(f"❌ Error creating instance of element '{element}'.")
                exit(1)
            pipe_parts.append((element, one_instance))
        return Pipeline(pipe_parts)

    @logger.catch
    def create_pipeline_training_info(self, pipe_details: dict, pipe: Pipeline) -> dict:
        """Create pipeline training info
        Args:
            pipe_details (dict): pipe details
            pipe (Pipeline): pipeline

        Returns:
            dict: pipeline training info
        """
        pipeline_training_info = {}
        for data_source, data_source_info in pipe_details.get("data_sources",{}).items():
            if data_source not in self.connections:
                logger.warning(f"⚠️ Datasource of type '{data_source}' is not implemented yet")
                continue
            if data_source == "influxdb":
                query = ""
                if isinstance(data_source_info, list):
                    query = "\n|> ".join(data_source_info).format(**pipe_details.get("steps",{}).get(pipe.steps[0][0],{}).get(data_source,{}))
                elif isinstance(data_source_info, dict):
                    data_source_info["buckets"] = self.get_valid_buckets(data_source_info.get("buckets"))
                    if len(data_source_info["buckets"]) > 1:
                        subqueries = []
                        for bucket in data_source_info["buckets"]:
                            query_params = {k:v for k,v in data_source_info.items() if k!="buckets"}
                            query_params["bucket"] = bucket
                            subquery = f"{bucket} = {self.connections[data_source]['connector'].compose_influx_query_from_dict(query_params)}"
                            subquery += f'\n|> map(fn: (r) => ({{ r with bucket: "{bucket}" }}))'
                            subqueries.append()
                        query = "\n".join(subqueries+[f"union(tables:[{','.join(data_source_info['buckets'])}])"])
                    else:
                        query_params = {k:v for k,v in data_source_info.items() if k!="buckets"}
                        query_params["bucket"] = data_source_info["buckets"][0]
                        query = self.connections[data_source]['connector'].compose_influx_query_from_dict(query_params)
                        query += f'\n|> map(fn: (r) => ({{ r with bucket: "{query_params["bucket"]}" }}))'
                elif isinstance(data_source_info, str):
                    query = data_source_info
                else:
                    logger.critical(f"❌ Definition of query for data source '{data_source}' using type '{type(data_source_info)}' is not allowed. Only 'list', 'dict' or 'str' allowed.")
                    exit(1)
            elif data_source == "postgresql":
                query = ""
                if isinstance(data_source_info, list):
                    query = "\n".join(data_source_info).format(**pipe_details.get("steps",{}).get(pipe.steps[0][0],{}).get(data_source,{}))
                elif isinstance(data_source_info, str):
                    query = data_source_info
                else:
                    logger.critial(f"❌ Definition of query for data source '{data_source}' using type '{type(data_source_info)}' is not allowed. Only 'list' or 'str' allowed.")
                    exit(1)
            elif data_source == "local":
                query = []
                if isinstance(data_source_info, list):
                    for folder in self.connections[data_source]:
                        for file in data_source_info:
                            filepath = os.path.join(folder,file)
                            if os.path.exists(filepath):
                                query.append(filepath)
                else:
                    logger.critial(f"❌ Definition of files for data source '{data_source}' using type '{type(data_source_info)}' is not allowed. Only 'str' allowed.")
                    exit(1)
            pipeline_training_info[data_source] = query
        return pipeline_training_info

    @logger.catch
    def create_pipelines(self) -> dict:
        """Create all pipes
        Returns:
            dict:pipelines
        """
        pipelines = {}
        for pipe_name, pipe_details in self.pipeline_config.items():
            pipe = self.create_one_pipeline(pipe_details.get("steps",{}))
            pipe_training_info = self.create_pipeline_training_info(pipe_details, pipe)
            pipelines[pipe_name] = {
                "pipe": pipe, 
                "training_info": pipe_training_info
            }
        logger.info(f"✅ Pipelines created successfully")
        return pipelines  

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
            module_name = f"{self.global_config.get('models_path')}.{class_elements[0]}"
            module = importlib.import_module(module_name)
            the_class = getattr(module, class_elements[1])
            instance = the_class(**class_attributes)
        except ModuleNotFoundError as e: 
            logger.error(f"❌ Error: The module'{class_elements[0]}', not found or other error in module: {e}")
        except AttributeError as e:
            logger.error(f"❌ Error: The class '{class_elements[1]}' not found: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        return instance


class PipelineExecutor(PipelineManager):

    @logger.catch
    def __init__(self, pipeline_config_file: str, total_processing: int = 4, generate_so: bool = True, save_in_joblib: bool = False):
        """Initializes the PipelineExecutor with configuration and processing options.

        Args:
            pipeline_config_file (str): Configuration file path.
            total_processing (int, optional): Number of processes to use. Defaults to 4.
            generate_so (bool, optional): Flag to generate shared object. Defaults to True.
            save_in_joblib (bool, optional): Flag to save in joblib format. Defaults to False.
        """
        super().__init__(pipeline_config_file)
        self.total_processing = total_processing
        self.generate_so = generate_so
        self.save_in_joblib = save_in_joblib
        
    @logger.catch
    def data_preparation(self, pipeline: Pipeline, training_info: dict) -> pd.DataFrame:
        """Prepare data for the pipeline execution.

        Args:
            pipeline (Pipeline): Pipeline
            training_info (dict): Pipeline data containing the queries and data origins

        Returns:
            pd.DataFrame: Data to fit the pipe
        """
        data = None
        try:
            for step_name, step_instance in pipeline.named_steps.items():
                if "MetaFuse" in [step_instance_parent.__name__ for step_instance_parent in step_instance.__class__.__bases__]:
                    data = step_instance.fuse_data_sources(
                        connections = self.connections,
                        training_info = training_info
                    )
        except Exception as err:
            logging.warning(f"Error preparing data for pipeline {pipeline}: {err}")
        else:
            if data is None:
                logger.warning("Training data is not available")
        finally:
            return data
        

    @logger.catch
    def pipes_executor(self, testing: bool = False):
        """Executes the pipeline tasks using multiprocessing.

        Args:
            testing (bool, optional): Flag to indicate if the execution is for testing purposes (fit and predict). Defaults to False.
        """
        with multiprocessing.Pool(processes=self.total_processing) as pool:
            for pipe_name, pipe_info in self.pipelines.items():
                logger.info(f"⚙️ Starting pipeline task '{pipe_name}'")
                data = self.data_preparation(
                    pipeline = pipe_info.get("pipe"),
                    training_info = pipe_info.get("training_info",{})
                )
                if data is None:
                    logger.warning("⚠️ Empty data, can't do training")
                    continue
                else:
                    pipe_core = {
                        "name": pipe_name, 
                        "pipe": pipe_info.get("pipe"), 
                        "training_info": pipe_info.get("training_info")
                    }
                    pool.apply_async(
                        lab_fit, 
                        args=(data, pipe_core, testing),
                        callback=self.task_handler,
                        error_callback=self.error_handler
                        )
                    logger.info(f"⚙️ Fitting pipeline task '{pipe_name}'")
            pool.close()
            pool.join()

    @logger.catch
    def task_handler(self, result):
        """Handles the result of the pipeline fitting process.

        Args:
            result: The result of the fitting process.
        """
        if result == "InsufficientDataError":
            logger.critical(f"❌ Not enough data to train")
        elif result == "KeyError":
            logger.critical(f"❌ The pipe is malformed, keys are missing")
        else:
            training_file = pipe_save(result, self.save_in_joblib)
            logger.success(f"✅ Pipe fitting successfully finished and stored in '{training_file}'")
            if self.generate_so:
                logger.info("⚙️ Creating shared object")
                self.launch_create_so(training_file)

    @logger.catch
    def error_handler(self, result):
        """Handles errors that occur during the fitting process.

        Args:
            result: The error result from the fitting process.
        """
        logger.error(f"❌ Error in fit {result}")

    @logger.catch
    def launch_create_so(self, model_path: str) -> None:
        """Launches the creation of a shared object.

        Args:
            model_path (str): Path to the model file.
        """
        create_so(model_path=model_path)
        logger.success(f"✅ Shared object created and stored in '{model_path}'")