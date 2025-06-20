'''
 # @ Project: AItea-Brain-Lite
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-03-28 13:29:53
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-05-24 20:48:06
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 '''


from loguru import logger
import os
import time
import pandas as pd
from typing import Union
from datetime import datetime


from influxdb_client import InfluxDBClient, BucketsApi
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.rest import ApiException
from influxdb_client.client.write_api import SYNCHRONOUS

import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore",MissingPivotFunction)

from utils.file_utils import get_configuration
from utils.logger_config import get_logger
logger = get_logger()

class InfluxDBConnector(object):
    """Class to create a connection with InfluxDB
    and manage buckets creation, buckets query
    and data insertion.
    """
    _instance = None

    @logger.catch()
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InfluxDBConnector, cls).__new__(cls)
        return cls._instance

    @logger.catch()
    def __init__(self):
        self.influx_client = None
        self.influx_client_writer = None
        self.am_bucket_api = None
        self.existence = True
        self.config_error = False 
        global_config = get_configuration()
        self.data_base_config = global_config.get("influxdb")
        self.load_configuration()
        self.queries = global_config.get("influx_queries")
        self.connection_status, self.client = self.connect()
        self.bucket_list = self.get_bucket_list()

    @logger.catch()
    def load_configuration(self,):
        """Method to load Influxdb configuration connection from
        global_config.json file
        """
        if self.data_base_config:
            self.token = self.data_base_config.get('token')
            self.host = self.data_base_config.get('host')
            self.port = self.data_base_config.get('port')
            self.org = self.data_base_config.get('org')
        else:
            self.config_error = True
            logger.error("Influxdb parameters for configuration are not defined in global config file. Check it again!")

    @logger.catch()
    def connect(self) -> tuple[bool, InfluxDBClient]:
        """Method that creates the InfluxDB connection
        """
        flag_connection = False
        if not self.config_error:
            try:
                url = "http://" + self.host + ":" + str(self.port)
                self.influx_client = InfluxDBClient(url=url, org=self.org, token=self.token, timeout=20000)
                self.am_bucket_api = BucketsApi(self.influx_client)
                self.influx_client_writer = self.influx_client.write_api(write_options=SYNCHRONOUS)
                time.sleep(3)
                health = self.influx_client.health()
                ping = self.influx_client.ping()
                msg = health.message
                status = health.status
                if ping:
                    flag_connection = True
                    logger.info(f"InfluxDB connection created successfully. InfluxDB client informs '{msg}' where the status is '{status}'")
                else:
                    logger.error(f"InfluxDB connection has not been created!. InfluxDB client informs '{msg}' and its status is '{status}'")
            except Exception as err:
                logger.error(f"An exception has arisen while creating the connection. Exception: {err}")
        else:
            logger.error("There is no connection data")
        return flag_connection, self.influx_client

    @logger.catch()
    def close(self) -> None:
        """Method to close InfluxDB connection
        """
        if self.influx_client:
            self.influx_client_writer.close()
            self.influx_client.close()
            logger.info("The connection has been closed")

    @logger.catch()
    def insert_data(self, bucket: str, point):
        """Method to insert data in influxdb

        Args:
            bucket (str): name of bucket where data will be stored
            point (Point): Point object from influxdbclient that contains dictionary with the data to be stored
        """
        if self.influx_client_writer:
            try:
                self.influx_client_writer.write(bucket=bucket, record=point, org=self.org)
                # logger.debug(f"Pointname content ({point}) has been inserted in '{bucket}' sucessfully")
            except InfluxDBError as err:
                self.exception_handler(err)
            except Exception as err:
                logger.error(f"An exception has arisen while trying to write the data. Exception: {err}")
        else:
            logger.error("There is no writer defined and also a problem with InfluxDB connection")
    
    @logger.catch()
    def insert_dataframe(self, bucket: str, dataframe: pd.DataFrame, tags: list, measurement: str, timestamp: str) -> None:
        logger.info(f"Inserting into bucket '{bucket}' a {dataframe.shape} pandas DataFrame with tags in columns '{tags}', measurement in column '{measurement}' and timestamps in column '{timestamp}")
        if self.influx_client_writer:
            try:
                logger.debug(f"Sneak peek of the dataframe to insert:\n{dataframe.head()}")
                self.influx_client_writer.write(
                    bucket=bucket, 
                    record=dataframe,
                    data_frame_measurement_name=measurement, 
                    data_frame_tag_columns=tags,
                    data_frame_timestamp_column=timestamp, 
                    org=self.org
                )
            except InfluxDBError as err:
                self.exception_handler(err)
            except Exception as err:
                logger.error(f"An exception has arisen while trying to write the data. Exception: {err}")
            else:
                logger.success("All rows have been successfully inserted.")
        else:
            logger.error("There is no writer defined and also a problem with InfluxDB connection")

    @logger.catch()
    def get_bucket_list(self) -> list:
        """Method to get all the available buckets

        Returns:
            list: Buckets list
        """
        bucket_list = list()
        if self.influx_client.ping():
            try:
                available_buckets = self.am_bucket_api.find_buckets(limit=100)
                for bucket in available_buckets.buckets:
                    bucket_name = bucket.name
                    if not bucket.name.startswith("_"):
                        bucket_list.append(bucket_name)                
            except ApiException as err:
                logger.error(f"There is an ApiException from InfluxDB trying to get buckets list. ApiException: {err}")
            except Exception as err:
                logger.error(f"There is an Exception from InfluxDB trying to get buckets list. ApiException: {err}")
    
        return bucket_list        

    @logger.catch()
    def bucket_creator(self, bucket_candidate: str) -> None:
        """Method to create a bucket

        Args:
            bucket_candidate (str): bucket name to verify its existence
        """
        if self.influx_client.ping():
            if bucket_candidate not in self.bucket_list:
                try:
                    self.am_bucket_api.create_bucket(bucket_name=bucket_candidate)
                    logger.success(f"A new bucket has been created: '{bucket_candidate}'")
                except ApiException as err:
                    logger.error(f"There is an ApiException from InfluxDB trying to create a bucket. ApiException: {err}")
            else:
                logger.warning(f"The bucket already exists and it will not be created.")

    @logger.catch()
    def exception_handler(self, err: Exception) -> None:
        """Method to manage InfluxDB Exceptions 
        in the writing process

        Args:
            err (Exxception): exception that has been arisen
        """
        status = err.response.status
        message = err.message
        err_content = f"Status: {status}, Message: {message}, Exception: {err}"
        if status == 400:
            logger.error(f"Malformed data to be inserted. Check the data structure. {err_content}")
        elif status == 401:
            logger.error(f"Unauthorized access to InfluxDB. Check database credentials (host, port, token, organization). {err_content}")
        elif status == 404:
            logger.error(f"Organization or bucket not found. {err_content}")
        elif status == 503:
            logger.error(f"Server is temporarily unavailable to accept writes. {err_content}")
        else:
            logger.error(f"There is a InfluxDB Exception. Please Review in detail. {err_content}")

    @logger.catch()
    def query(self, query: str, schema: bool = False, pandas: bool = False, stream: bool = False) -> Union[list, pd.DataFrame]:
        """Make a influx simple query

        Args:
            query (str): Query string
            schema (boolean): Only schema return

        Returns:
            Union[list, pd.DataFrame]: All records in list or in a dataframe
        """
        time_i = time.time()
        logger.info("Starting the search and transformation")
        query_api = self.influx_client.query_api()
        if pandas:
            results = self._query_pandas(query_api, query, stream)
            logger.info(f"Finishing search and finishing the transformation to pandas in {(time.time() - time_i):.3f} seconds")
        else:
            results = self._query_list(query_api, query, schema)
            logger.info(f"Finishing search and finishing the transformation to list in {(time.time() - time_i):.3f} seconds")
        return results


    @logger.catch()
    def _query_pandas(self, query_api, query: str, stream: bool) -> pd.DataFrame:
        results = None
        try:
            if stream:
                result = query_api.query_data_frame_stream(query=query) 
            else:
                result = query_api.query_data_frame(query=query)
        except Exception as err:
            logger.error(f"Influx query exception: {err}")
        else:
            if isinstance(result, list):
                results = pd.concat(result)
            else:
                results = result
        return results

    @logger.catch()
    def _query_list(self, query_api, query: str, schema: bool = False) -> list:
        results = list()
        try:
            result = query_api.query_data_frame(query=query)
        except Exception as err:
            logger.error(f"Influx query exception {err}")
            result = []
        for table in result:
            time.sleep(10)
            for record in table.records:
                if schema:
                    results.append(record.get_value())
                else:
                    results.append(
                        (record.get_field(), record.get_value(), record.get_time())
                    )
        return results
    
    @logger.catch()
    def request_query(self, query_dict: dict, pandas: bool = True, stream=False):
        data_answer = None
        influx_query = self.compose_influx_query_from_dict(arguments_dict=query_dict)
        if influx_query:
            logger.info(f"Requesting results with the following influx query:{influx_query}")
            data_answer = self.query(query=influx_query, pandas=pandas, stream=stream)
        else:
            logger.warning(f"No query generated: no results retrieved!")
        return data_answer

    @logger.catch()
    def compose_influx_query_from_dict(self, arguments_dict: dict) -> str:
        """Compose a influx query (flux language)

        Args:
            arguments_dict (dict): Dictionary with labels, time ranges, measurements and bucket details.  

        Returns:
            str: Flux query
        """
        query = ""
        for key, value in arguments_dict.items():
            try:
                if key == "range":
                    value = self.to_unix_time(value)
                if isinstance(value, list):
                    query += "\n|> filter(fn: (r) =>"
                    size_list = len(value)
                    for pos, one_value in enumerate(value):
                        q = self.queries[key].replace("|> filter(fn: (r) =>", "")
                        q = q.replace(")", "")
                        one_template = q.format(**one_value)
                        query += one_template
                        if pos < size_list - 1:
                            query += " or "
                    query += ")"
                else:
                    one_template = self.queries[key].format(**value)
                    query += "\n" + one_template
            except Exception as err:
                logger.warning(f"There was an error while composing influx query from dict for key '{key}': {err}")
        return query
    
    @logger.catch()
    def to_unix_time(self, range: dict):
        start_unix = '0'
        stop_unix = '0'
        start = range.get("start")
        stop = range.get("stop")
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")
            stop_dt = datetime.strptime(stop, "%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception as err:
            logger.error(f"The date format is not correct {err}")
        else:
            start_unix = int(start_dt.timestamp())
            stop_unix = int(stop_dt.timestamp())
        return {"start": start_unix, "stop": stop_unix}
