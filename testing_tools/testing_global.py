'''
 # @ Author: Héctor Bermúdez Castro <hector.bermudez@aitea.tech>
 # @ Create Time: 2025-06-20
 # @ Modified by: Héctor Bermúdez Castro
 # @ Modified time: 2025-06-20
 # @ Project: Aitea Building Lab
 # @ Description: Analytic template
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from dotenv import load_dotenv
import os

from utils.file_utils import get_configuration
from utils.logger_config import get_logger

from database_tools.influxdb_connector import InfluxDBConnector
from testing_tools.testing_influxdb import generate_testing_data, upload_testing_data
from pipelines.pipeline_executor import PipelineExecutor

import pandas as pd

if __name__ == "__main__":

    try:
        # Setting logger
        logger = get_logger()
        logger.debug("Aitea Building Lab test started")

        # Configuration
        load_dotenv()
        config_json = get_configuration()
        testing_conf = get_configuration().get("testing")
        pipe_plan_path = testing_conf.get("pipe_plan_path")
        logger.success(f"Configuration loaded successfully from {os.getenv('CONFIG_PATH')}")
        
        # Creating testing data
        path = testing_conf.get("data").get("path")
        generate_testing_data(testing_conf.get("data"))
        testing_df = pd.read_csv(path, skiprows=3).drop(columns=["Unnamed: 0"])
        logger.success(f"Testing data correctly generated. Sneak peek:\n{testing_df.head()}")

        # Uploading testing data
        influxdb = InfluxDBConnector()
        upload_testing_data(influxdb, testing_conf, testing_df)
        
        # Retrieving testing data
        data = influxdb.request_query(query_dict=testing_conf.get("query"), pandas=True)
        influxdb.close()
        logger.info(f"Testing data correctly retrieved from database. Sneak peek:\n{data.head()}")

        # Testing pipelines generating model .pkl and library .so
        pipe = PipelineExecutor(pipe_plan_path, generate_so=True, save_in_joblib=False)
        pipe.pipes_executor(testing=False)
        logger.success("Pipeline execution was successful")

    except Exception as err:
        logger.error(f"Error found when running test: {err}")
    else:
        logger.success("Aitea Building Lab test was a success!")
    finally:
        logger.debug("Aitea Building Lab test completed")