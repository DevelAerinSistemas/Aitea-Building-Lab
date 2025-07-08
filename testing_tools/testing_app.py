'''
 # @ Author: Héctor Bermúdez Castro <hector.bermudez@aitea.tech>
 # @ Create Time: 2025-06-20
 # @ Modified by: Héctor Bermúdez Castro
 # @ Modified time: 2025-06-20
 # @ Project: Aitea Building Lab
 # @ Description: Testing app
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from dotenv import load_dotenv
import os

from utils.file_utils import load_json_file
from utils.logger_config import get_logger

try:
    from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
    from aitea_connectors.connectors.postgresql_connector import PostgreSQLConnector
    AITEA_CONNECTORS = True
except ImportError:
    logger.warning(f"⚠️ Aitea Connectors are not available. Uncomplete functionality: only local files as a valid data source.")
    AITEA_CONNECTORS = False

from pipelines.pipeline_executor import PipelineExecutor

import pandas as pd

if __name__ == "__main__":

    try:
        # Setting logger
        logger = get_logger()
        logger.info("🚀 Aitea Building Lab General test launched!")

        # Configuration
        load_dotenv()
        config_path = os.getenv("CONFIG_PATH")
        config_json = load_json_file(config_path)
        logger.success(f"✅ Configuration loaded successfully from {os.getenv('CONFIG_PATH')}")
        pipe_plan_path = config_json.get("pipe_plan_path")
        logger.info(f"💬 Using pipe schedule from '{pipe_plan_path}'")

        # Testing pipelines generating model .pkl and library .so
        pipe = PipelineExecutor(pipe_plan_path, generate_so=True, save_in_joblib=False)
        pipe.pipes_executor(testing=False)
        logger.success("✅ Pipeline execution was successful")

    except Exception as err:
        logger.error(f"❌ Error found when running test: {err}")
    else:
        logger.success("🎉 Aitea Building Lab General test was a success!")
    finally:
        logger.success("✅ Aitea Building Lab General test completed")