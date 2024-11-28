'''
 # @ Author: Jose Masache Cevallos <jose.masache@aerin.es>
 # @ Create Time: 2024-07-23 09:42:57
 # @ Modified time: 2024-07-23 09:43:01
 # @ Project: aitea building lab
 # @ Description: 
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


from utils.file_utils import get_configuration


class InfluxDBConnector(object):
    """Class to create a connection with InfluxDB
    and manage buckets creation, buckets query
    and data insertion.
    """
    @logger.catch()
    def __init__(self):
        self.influx_client = None
        self.influx_client_writer = None
        self.am_bucket_api = None
        self.existence = True
        self.config_error = False
        connection_config = get_configuration(
            global_configuration="config/global_config.json", section="connections")
        self.data_base_config = connection_config.get("influxdb")
        self.load_configuration()
        self.queries = get_configuration(
            global_configuration="config/influx_q.json")

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
            logger.error(
                "[AM][InfluxDB][Connector] - Influxdb parameters for configuration are not defined in global config file. Check it again!")

    @logger.catch()
    def connect(self, flag_disconnection):
        """Method that creates the InfluxDB connection
        """
        flag_connection = False
        if not self.config_error:
            try:
                url = "http://" + self.host + ":" + str(self.port)
                self.influx_client = InfluxDBClient(
                    url=url, org=self.org, token=self.token, timeout=20000)
                self.am_bucket_api = BucketsApi(self.influx_client)
                self.influx_client_writer = self.influx_client.write_api(
                    write_options=SYNCHRONOUS)
                time.sleep(3)
                health = self.influx_client.health()
                ping = self.influx_client.ping()
                msg = health.message
                status = health.status
                if ping:
                    flag_connection = True
                    if flag_disconnection and flag_connection:
                        logger.info(
                            "[AM][InfluxDB][Connector] - InfluxDB Connection has been restablished!")
                    logger.info(
                        f"[AM][InfluxDB][Connector] - InfluxDB connection created successfully. InfluxDB client informs '{msg}' where the status is '{status}'")

                else:
                    logger.error(
                        f"[AM][InfluxDB][Connector] - InfluxDB connection has not been created!. InfluxDB client informs '{msg}' and its status is '{status}'")
                    if flag_disconnection and not flag_connection:
                        logger.error(
                            "[AM][InfluxDB][Connector] - Trying to reconnect with InfluxDB")
            except Exception as err:
                logger.error(
                    f"[AM][InfluxDB][Connector] - An exception has arisen while creating the connection. Exception: {err}")
        else:
            logger.error(
                "[AM][InfluxDB][Connector] - There is no connection data")
        return flag_connection, self.influx_client

    @logger.catch()
    def close(self) -> None:
        """Method to close InfluxDB connection
        """
        if self.influx_client:
            self.influx_client_writer.close()
            self.influx_client.close()
            logger.info(
                "[AM][InfluxDB][Connector] - The connection has been closed")

    @logger.catch()
    def insert_data(self, bucket: str, point):
        """Method to insert data in influxdb

        Args:
            bucket (str): name of bucket where data will be stored
            point (Point): Point object from influxdbclient that contains
            dictionary with the data to be stored
        """
        if self.influx_client_writer:
            try:
                self.influx_client_writer.write(
                    bucket=bucket, record=point, org=self.org)
                # logger.debug(
                #     f"[AM][InfluxDB][Connector] - Pointname content ({point}) has been inserted in '{bucket}' sucessfully")
            except InfluxDBError as err:
                self.exception_handler(err)
            except Exception as err:
                logger.error(
                    f"[AM][InfluxDB][Connector] - An exception has arisen while trying to write the data. Exception: {err}")
        else:
            logger.error(
                "[AM][InfluxDB][Connector] - There is no writer defined and also a problem with InfluxDB connection")

    @logger.catch()
    def insert_dataframe(self, bucket: str, dataframe: pd.DataFrame, tags: list, measurement: str):
        if self.influx_client_writer:
            try:
                self.influx_client_writer.write(
                    bucket=bucket, record=dataframe, data_frame_measurement_name=measurement, data_frame_tag_columns=tags, org=self.org)
            except InfluxDBError as err:
                self.exception_handler(err)
            except Exception as err:
                logger.error(
                    f"[AM][InfluxDB][Connector] - An exception has arisen while trying to write the data. Exception: {err}")
        else:
            logger.error(
                "[AM][InfluxDB][Connector] - There is no writer defined and also a problem with InfluxDB connection")

    @logger.catch()
    def get_bucket_list(self) -> list:
        """Method to get all the available buckets

        Returns:
            list: Buckets list
        """
        bucket_list = list()
        if self.influx_client.ping():
            try:
                available_buckets = self.am_bucket_api.find_buckets().buckets
                for bucket in available_buckets:
                    bucket_list.append(bucket.name)
            except ApiException as err:
                logger.error(
                    f"[AM][InfluxDB][Insertion] - There is an ApiException from InfluxDB trying to get buckets list. ApiException: {err}")
            except Exception as err:
                logger.error(
                    f"[AM][InfluxDB][Insertion] - There is an Exception from InfluxDB trying to get buckets list. ApiException: {err}")
        return bucket_list

    @logger.catch()
    def bucket_creator(self, bucket_candidate: str):
        """Method to create a bucket

        Args:
            bucket_candidate (str): bucket name to verify its existence
        """
        if self.influx_client.ping():
            self.bucket_reviewer(bucket_candidate)
            if not self.bucket_reviewer(bucket_candidate):
                try:
                    self.am_bucket_api.create_bucket(
                        bucket_name=bucket_candidate)
                    logger.info(
                        f"[AM][InfluxDB][Connector] - A new bucket has been created: {bucket_candidate}")
                except ApiException as err:
                    logger.error(
                        f"[AM][InfluxDB][Insertion] - There is an ApiException from InfluxDB trying to create a bucket. ApiException: {err}")
            else:
                logger.warning(
                    f"[AM][InfluxDB][Insertion] - The bucket exists and will not be created.")

    @logger.catch()
    def bucket_reviewer(self, bucket_candidate: str):
        """Method to check the bucket existence 

        Args:
            bucket_candidate (str): bucket to be verified
        """
        existence = False
        bucket_list = self.get_bucket_list()
        if bucket_list is not None:
            if bucket_candidate in bucket_list:
                existence = True
        return existence

    @logger.catch()
    def exception_handler(self, err: Exception):
        """Method to manage InfluxDB Exceptions 
        in the writing process

        Args:
            err (Exception): exception that has been arisen
        """
        status = err.response.status
        message = err.message
        err_content = f"Status: {status}, Message: {message}, Exception: {err}"
        if status == 400:
            logger.error(
                f"[AM][InfluxDB][Connector] - Malformed data to be inserted. Check the data structure. {err_content}")
        elif status == 401:
            logger.error(
                f"[AM][InfluxDB][Connector] - Unauthorized access to InfluxDB. Check database credentials (host, port, token, organization). {err_content}")
        elif status == 404:
            logger.error(
                f"[AM][InfluxDB][Connector] - Organization or bucket not found. {err_content}")
        elif status == 503:
            logger.error(
                f"[AM][InfluxDB][Connector] - Server is temporarily unavailable to accept writes. {err_content}")
        else:
            logger.error(
                f"[AM][InfluxDB][Connector] - There is a InfluxDB Exception. Please Review in detail. {err_content}")

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
            logger.info(
                f"Finishing search and finishing the transformation to pandas in {time.time() - time_i}")
        else:
            results = self._query_list(query_api, query, schema)
            logger.info(
                f"Finishing search and finishing the transformation to list in {time.time() - time_i}")
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
            logger.error(f"Influx query exception {err}")
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
                        (record.get_field(), record.get_value(), record.get_time()))
        return results

    @logger.catch()
    def request_query(self, query_dict: dict, pandas: bool = True, stream=False):
        influx_query = self.compose_influx_query_from_dict(
            arguments_dict=query_dict)
        logger.info(
            f"Requesting a query with the following influx query language {influx_query}")
        data_answer = self.query(
            query=influx_query, pandas=pandas, stream=stream)
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


if __name__ == "__main__":
    query = {"bucket": {"bucket": "alfonso_xii_62"}, "range": {"start": "2024-09-1T00:00:00.000Z", "stop": "2024-09-15T05:30:00.000Z"},
             "filter_measurement": {"measurement": "climatization"}, "filter_field": [{"field": "room_temperature"}, {"field": "setpoint_temperature"}, {"field": "general_condition"}, {"field": "setpoint_temperature"}], "tag_is": {"tag_name": "type", "tag_value": "air_conditioning"}, "window_aggregation": {"every": "5m", "function": "mean", "create_empty": "false"}, "fill": {"filltype": "usePrevious", "true_false": "true"}}
    influx = InfluxDBConnector()
    _, _ = influx.connect(True)
    data = influx.request_query(query_dict=query, pandas=True)
    print(data)
    # influx.close()
