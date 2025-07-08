'''
 # @ Project: AItea-Brain-Lite
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-05-27 19:19:12
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-05-27 20:59:04
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 '''
 
import importlib
import datetime
import pytz
import json
import re
import pandas as pd

from dotenv import load_dotenv
load_dotenv()
import os

from utils.logger_config import get_logger
logger = get_logger()
from utils.file_utils import load_json_file

try:
    from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
    from aitea_connectors.connectors.postgresql_connector import PostgreSQLConnector
    AITEA_CONNECTORS = True
except ImportError:
    logger.warning(f"⚠️ Aitea Connectors are not available. Uncomplete functionality: only local files as a valid data source.")
    AITEA_CONNECTORS = False

class SOLibraryLoader:
    def __init__(self, name: str):
        """
        Initializes the SOLibraryLoader with the name of the .so library.

        Args:
            name (str): The name of the .so library (without the .so extension).

        Returns:
            tuple: A tuple containing (results, result_matrix).
        """
        if AITEA_CONNECTORS:
            self.influxdb_conn = InfluxDBConnector()
            influxdb_conn_status, influxdb_conn_client = self.influxdb_conn.connect()
            self.postgresql_conn = PostgreSQLConnector()
            postgresql_conn_status, postgresql_conn_client = self.postgresql_conn.connect()
        else:
            self.influxdb_conn = None
            self.postgresql_conn = None
        self.name = name
        self.exec = self._load_library()
    
    def testing_predict_with_file(self, file_path: str) -> tuple:
        """
        Loads data from a local file, runs prediction, and returns results.
        
        Args:
            file_path (str): The absolute path to the data file.
            
        Returns:
            tuple: A tuple containing (results, result_matrix).
        """
        prediction_dict = dict()
        results_dict = dict()
        if self.exec is not None:
            data_df = pd.DataFrame()
            if file_path.endswith('.csv'):
                data_df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            elif file_path.endswith(".json"):
                data_df = pd.read_json(datafile)
            elif file_path.endswith(".xlsx") or file_path.endswith(".ods"):
                data_df = pd.read_excel(datafile)
            elif file_path.endswith('.parquet'):
                data_df = pd.read_parquet(file_path)
            else:
                source_type = file_path.split(".")[-1]
                logger.warning(f"⚠️ Reading data from source of type '{source_type}' not implemented yet")
                raise ValueError(f"Unsupported file type for '{file_path}'")
            if not data_df.empty:
                try: 
                    prediction_dict["demo"] = self.exec.predict(bucket_data)
                    results_dict["demo"] = self.exec.get_results()
                    logger.success(f"✅ Prediction complete for file: '{os.path.basename(file_path)}'")
                except Exceptions as err:
                    logger.error(f"❌ Uncontrolled error: {err}")
                    raise Exception(f"Uncontrolled error: {err}")
            else:
                logger.warning(f"⚠️ Empty predicting data")
                raise ValueError(f"Empty predicting datafile")
        else:
            logger.error("❌ Executor class instance is not initialized.")
        return prediction_dict, results_dict

    def testing_predict_with_postgresql(self, start_datetime: str, stop_datetime: str) -> tuple:
        """
        Predict new data using a library and two datetimes provided by GUI and data from PostgreSQL

        Args:
            start_datetime (int): The new integer timestamp for the 'start' value.
            stop_datetime (int): The new integer timestamp for the 'stop' value.
        """
        prediction_dict = dict()
        results_dict = dict()
        if self.exec is not None:
            training_info = self.exec.get_training_info()
            if AITEA_CONNECTORS and training_info.get("postgresql"):
                    postgresql_query = self._compose_sql_query(
                        query = training_info.get("postgresql"),
                        start_datetime = start_datetime,
                        stop_datetime = stop_datetime
                    )
                    data = self.postgresql_conn.query_to_df(query = postgresql_query)
                    logger.info(f"💬 Model Info: {self.exec.get_info()}")
                    if data is not None and not data.empty:
                        one_prediction = self.exec.predict(data)
                        #TODO Change whenever PostgreSQL integration is complete
                        prediction_dict["demo"] = one_prediction 
                        results_dict["demo"] = self.exec.get_results()
            else:
                logger.warning(f"⚠️ Aitea Connectors are not available. InfluxDB not available as a datasource.")
        else:
            logger.error("❌ Executor class instance is not initialized.")
        return prediction_dict, results_dict

    def _compose_sql_query(self, query: str, start_datetime: str, stop_datetime: str) -> str:
        """
        Substitutes the start and stop values in the range() function of a SQL query.

        Args:
            query (str): The original SQL query string.
            start_datetime (int): The new integer timestamp for the 'start' value.
            stop_datetime (int): The new integer timestamp for the 'stop' value.

        Returns:
            str: The new SQL query with the updated start and stop times.
        """ 
        #TODO Change whenever PostgreSQL integration is complete
        new_query = query
        return new_query

    def testing_predict_with_influx(self, start_datetime: str, stop_datetime: str) -> tuple:
        """
        Predict new data using a library and two datetimes provided by GUI and data from InfluxDB

        Args:
            start_datetime (int): The new integer timestamp for the 'start' value.
            stop_datetime (int): The new integer timestamp for the 'stop' value.
        """
        prediction_dict = dict()
        results_dict = dict()
        if self.exec is not None:
            training_info = self.exec.get_training_info()
            if AITEA_CONNECTORS:
                if training_info.get("influxdb"):
                    influxdb_query = self._compose_flux_query(
                        query = training_info.get("influxdb"),
                        start_datetime = start_datetime,
                        stop_datetime = stop_datetime
                    )
                    data = self.influxdb_conn.query(query=influxdb_query, pandas=True, stream=False)
                    logger.info(f"💬 Model Info: {self.exec.get_info()}")
                    if data is not None and not data.empty:
                        for bucket in data["bucket"].unique():
                            bucket_data = data[data["bucket"]==bucket].reset_index(drop=True)
                            one_prediction = self.exec.predict(bucket_data)
                            prediction_dict[bucket] = one_prediction
                            results_dict[bucket] = self.exec.get_results()
            else:
                logger.warning(f"⚠️ Aitea Connectors are not available. InfluxDB not available as a datasource.")
        else:
            logger.error("❌ Executor class instance is not initialized.")
        return prediction_dict, results_dict
    
    def _compose_flux_query(self, query: str, start_datetime: str, stop_datetime: str) -> str:
        """
        Substitutes the start and stop values in the range() function of a Flux query.

        Args:
            query (str): The original Flux query string.
            start_datetime (int): The new integer timestamp for the 'start' value.
            stop_datetime (int): The new integer timestamp for the 'stop' value.

        Returns:
            str: The new Flux query with the updated start and stop times.
        """ 
        new_query = re.sub(
            pattern=r"(start\s*:\s*)\d+", 
            repl=f'\\1time(v:"{start_datetime}")', 
            string=query
        )
        new_query = re.sub(
            pattern=r"(stop\s*:\s*)\d+", 
            repl=f'\\1time(v:"{stop_datetime}")',
            string=new_query
        )
        return new_query

    def analyze_lib(self):
        """Analyzes the loaded library and retrieves relevant information.
        """
        if self.exec:
            memory_data = self.exec.get_all_attributes()
            for key, matrix in memory_data.items():
                for key, one in matrix.items():
                    logger.info(f"Memory data: {key} - {one}")
                    if isinstance(one, pd.DataFrame):
                        logger.info(f"DataFrame shape: {one.shape}")
                        logger.info(f"DataFrame columns: {one.columns.tolist()}")
                        logger.info(f"NaN values count:\n{one.isnull().sum()}")
                        for col in one.columns:
                            logger.info(f"💬 Column '{col}' unique values: {one[col].unique()}")  # Display first 5 unique values
        else:
            logger.warning("⚠️ No library loaded, cannot analyze.")
    
    def get_info(self):
        """
        Retrieves information about the loaded library.

        Returns:
            dict: Information about the library.
        """
        info = {}
        if self.exec is not None:
            info = self.exec.get_info()
           
        else:
            logger.error("❌ Executor class instance is not initialized.")
        return info
    
    def check_query(self):
        """
        Checks the query and logs the result.
        """
        if self.exec is not None:
            query = self.exec.get_training_info()
            logger.info(f"🔍 Checked query:\n{training_info}")
        else:
            logger.error("❌ Executor class instance is not initialized.")
    
    def _load_library(self):
        """
        Loads the .so library using importlib.
        Returns:
            module: The loaded module, or None if loading fails.
        """
        module_name = f"lib.{self.name}"
        executor_class_instance = None
        try:
            module = importlib.import_module(module_name)
            ExecutorClass = getattr(module, "PipeExecutor", None)
            executor_class_instance = ExecutorClass()
        except ImportError as e:
            logger.error(f"❌ Failed to load library {module_name}: {e}")
            raise FileNotFoundError(f"Library {module_name} not found.")
        else:
            logger.info(f"✅ Library {module_name} loaded successfully.")
            logger.info(f"🔍 Executor class found: {ExecutorClass.__name__ if ExecutorClass else 'None'}")
            logger.info(f"💬 Library info: {ExecutorClass.__doc__}")
            logger.info(f"💬 Library info: {executor_class_instance.get_info()}")
        finally:
            return executor_class_instance