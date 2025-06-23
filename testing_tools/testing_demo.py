'''
 # @ Author: Héctor Bermúdez Castro <hector.bermudez@aitea.tech>
 # @ Create Time: 2025-06-20
 # @ Modified by: Héctor Bermúdez Castro
 # @ Modified time: 2025-06-20
 # @ Project: Aitea Building Lab
 # @ Description: Testing app with testing configuration
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from dotenv import load_dotenv
import os

from utils.file_utils import get_configuration
from utils.logger_config import get_logger

from database_tools.influxdb_connector import InfluxDBConnector
from testing_tools.testing_influxdb import generate_demo_data, upload_demo_data
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
        demo_conf = get_configuration().get("demo")
        pipe_plan_path = demo_conf.get("pipe_plan_path")
        logger.success(f"Configuration loaded successfully from {os.getenv('CONFIG_PATH')}")
        
        # Creating demo data
        path = demo_conf.get("data").get("path")
        generate_demo_data(demo_conf.get("data"))
        demo_df = pd.read_csv(path, skiprows=3).drop(columns=["Unnamed: 0"])
        logger.success(f"Demo data correctly generated. Sneak peek:\n{demo_df.head()}")

        # Uploading demo data
        influxdb = InfluxDBConnector()
        upload_demo_data(influxdb, demo_conf, demo_df)
        
        # Retrieving demo data
        data = influxdb.request_query(query_dict=demo_conf.get("query"), pandas=True)
        influxdb.close()
        logger.info(f"Demo data correctly retrieved from database. Sneak peek:\n{data.head()}")

        # Demo pipelines generating model .pkl and library .so
        pipe = PipelineExecutor(pipe_plan_path, generate_so=True, save_in_joblib=False)
        pipe.pipes_executor(testing=False)
        logger.success("Pipeline execution was successful")

    except Exception as err:
        logger.error(f"Error found when running test: {err}")
    else:
        logger.success("Aitea Building Lab test was a success!")
    finally:
        logger.debug("Aitea Building Lab test completed")