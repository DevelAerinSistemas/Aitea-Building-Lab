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
 
 
from database_tools.influxdb_connector import InfluxDBConnector
from utils.file_utils import load_json_file


import importlib
from loguru import logger
import datetime
import pytz
import json
import pandas as pd

class SOLibraryLoader:
    def __init__(self, name: str):
        """
        Initializes the SOLibraryLoader with the name of the .so library.

        Args:
            name (str): The name of the .so library (without the .so extension).
        """
        self.influxdb_conn = InfluxDBConnector()
        influxdb_conn_status, influxdb_conn_client = self.influxdb_conn.connect()
        self.name = name
        self.exec = None
        self.exec = self._load_library()
    
    def testing(self, stop_data: str, start_data: str) -> dict:
        """
        Placeholder for a testing method that you will implement.
        """
        results_dict = dict()
        if self.exec is not None:
            query = self.exec.get_query()
            # Extraemos la query
            prediction_dict = dict()
            composed_query_list = self._compose_query(query, stop_data, start_data)
            for one_query in composed_query_list:
                one_query_copy = one_query.copy()
                query = self.influxdb_conn.compose_influx_query_from_dict(one_query_copy)
                data = self.influxdb_conn.query(query=query, pandas=True, stream=False)
                logger.info(f"Model Info: {self.exec.get_info()}")
                if data is not None and not data.empty:
                    bucket = one_query.get("bucket", {}).get("bucket", "default_bucket")
                    data.loc[:, "bucket"] = bucket
                    # Se ejecuta la predicción para cada edificio
                    one_prediction = self.exec.predict(data)
                    prediction_dict[bucket] = one_prediction
                    results_dict[bucket] = self.exec.get_results()
        else:
            logger.error("Executor class instance is not initialized.")
        
        return prediction_dict, results_dict
            
    def _compose_query(self, query: str, end_time: str, start_time: str) -> str:
        """
        Composes a query string with the provided start and end times.

        Args:
            query (str): The base query string.
            start_time (str): The start time in ISO format.
            end_time (str): The end time in ISO format.

        Returns:
            str: The composed query string.
        """
        queries_list = list()
        buckets_list = query.get("bucket", {}).get("bucket", [])
        bucket_query = query
        for one_bucket in buckets_list:
            one_bucket_query = bucket_query.copy()
            one_bucket_query.update({"bucket": {"bucket": one_bucket}})
            one_bucket_query.update({"range": {"start": start_time, "stop": end_time}})
            logger.info(f"Updated bucket query: {one_bucket_query}")
            queries_list.append(one_bucket_query)
        return queries_list
    
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
                            logger.info(f"Column '{col}' unique values: {one[col].unique()}")  # Display first 5 unique values
        else:
            logger.warning("No library loaded, cannot analyze.")
    
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
            logger.error("Executor class instance is not initialized.")
        return info
    
    def check_query(self):
        """
        Checks the query and logs the result.
        """
        if self.exec is not None:
            query = self.exec.get_query()
            logger.info(f"Checked query: {query}")
        else:
            logger.error("Executor class instance is not initialized.")
    
    def _load_library(self):
       """
       Loads the .so library using importlib.
       Returns:
           module: The loaded module, or None if loading fails.
       """
       module_name = f"lib.{self.name}"
       try:
           module = importlib.import_module(module_name)
           ExecutorClass = getattr(module, "PipeExecutor", None)
           executor_class_instance = ExecutorClass()
          
       except ImportError as e:
           logger.error(f"Failed to load library {module_name}: {e}")
           raise FileNotFoundError(f"Library {module_name} not found.")
       else:
           logger.info(f"Library {module_name} loaded successfully.")
           logger.info(f"Executor class found: {ExecutorClass.__name__ if ExecutorClass else 'None'}")
           logger.info(f"Library info: {ExecutorClass.__doc__}")
           logger.info(f"Library info: {executor_class_instance.get_info()}")
           return executor_class_instance
    

if __name__ == '__main__':
    for a in range(2):
    # Example usage:
        try:
            loader = SOLibraryLoader("pmvppd_analysis") 
        except FileNotFoundError as e:
            logger.error(e)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        else:
            loader.check_query()
            start_time = datetime.datetime(2025, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
            stop_time = datetime.datetime(2025, 1, 2, 23, 59, 59, tzinfo=pytz.UTC)
            loader.testing(stop_data=start_time, start_data=stop_time)