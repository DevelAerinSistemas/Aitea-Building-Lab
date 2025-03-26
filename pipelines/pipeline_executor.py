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

    def create_pipelines(self):
        """Create all pipes
        """
        if self.configuration:
            for pipe_name, pipe_values in self.configuration.items():
                pipe = self.create_one_pipeline(pipe_values)
                query = pipe_values.get("training_query")
                freq_info = pipe_values.get("freq_info")
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
            pipe_parts.append((element, one_instance))
        return Pipeline(pipe_parts)

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
        except ModuleNotFoundError:
            logger.error(f"Error: The module'{class_elements[0]}', not found")
        except AttributeError:
            logger.error(f"Error: The class '{class_elements[0]}' not found.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        return instance


class PipelineExecutor(PipelineManager, ):
    def __init__(self, configuration_definition_file: str, total_processing: int = 4, create_so: bool = True):
        super().__init__(configuration_definition_file)
        self.total_processing = total_processing
        self.create_pipelines()
        data_conection = load_json_file(os.getenv("INFLUX_CONNECTION"))
        self.create_so = create_so
        for name, pipes_data in self.pipes.items():
            influx = InfluxDBConnector(data_conection)
            influx.load_configuration()
            _, _ = influx.connect(True)
            pipes_data["connection"] = influx

    def data_preparation(self, pipe_data: dict) -> pd.DataFrame:
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
                total_dataframe = pd.concat(dataframe_list)
        return total_dataframe

    def pipes_executor(self):
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
                    pool.apply_async(lab_fit, args=(data, pipe_core,),
                                     callback=self.task_handler,
                                     error_callback=self.error_handler)
                    logger.info(f"Geting fit to {pipe_name}")
            pool.close()
            pool.join()

    def task_handler(self, result):
        if result == "InsufficientDataError":
            logger.critical(f" Not enough data to train")
        elif result == "KeyError":
            logger.critical(f" The pipe is malformed, keys are missing")
        else:
            pipe_save(result)
            logger.info(f" End pipe fit. It is saved {result}")
            if self.create_so:
                logger.info("Creating shared object")
                self.create_so()

    def error_handler(self, result):
        logger.error(f"Error in fit {result}")

    def create_so(self):
        command = [
            "/opt/VirtualEnv/virtualAiteaBuildingLab/bin/nuitka",
            "--module", "execution/executor.py",
            "--include-package=models_warehouse",
            "--include-package=metaclass",
            "--include-package=utils",
            "--show-modules",
            "--output-dir=lib",
            "--remove-output"
        ]
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.info(f" Output: {result.stdout}")
            if result.stderr:
                logger.error(f"Error Output: {result.stderr}")
        except  subprocess.CalledProcessError as err:
            logger.error(f"Error Output: {err}")

        



if __name__ == "__main__":
    pipe = PipelineExecutor("pipes_schedules/pipe_plan.json", create_so=False)
    pipe.pipes_executor()
