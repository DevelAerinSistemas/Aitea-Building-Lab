#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: influxdb_connector.py
# Project: Aitea-Brain
# File Created: Thursday, 7th March 2024 11:23:09 pm
# Author: Jose Masache Cevallos (jose.masache@aitea.tech)
# Version: 1.0.0
# -----
# Last Modified: Monday, 4th November 2024 9:29:34 pm
# Modified By: Jose Masache Cevallos
# -----
# Copyright (c) 2024 - 2024 Aitea Tech S. L. copying, distribution or modification not authorised in writing is prohibited.
###


from utils.file_utils import get_influx_queries


from datetime import datetime, timezone
import pandas as pd
import time
from typing import Union
from influxdb_client import InfluxDBClient, BucketsApi, Point, WriteOptions
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.rest import ApiException
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.flux_table import FluxStructureEncoder
import json


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
    def __init__(self, configuration):
        self.influx_client = None
        self.influx_client_writer = None
        self.am_bucket_api = None
        self.existence = True
        self.config_error = False
        self.global_config = configuration
        self.status_ok_connection = False
        self.load_configuration()
        self.queries = get_influx_queries()

    @logger.catch()
    def load_configuration(self,):
        """Method to load Influxdb configuration connection from
        global_config.json file
        """
        if self.global_config:
            try:
                self.influxdb_configuration = self.global_config.get(
                    'influxdb')
                if self.influxdb_configuration is not None:
                    self.token = self.influxdb_configuration.get('token')
                    self.host = self.influxdb_configuration.get('host')
                    self.port = self.influxdb_configuration.get('port')
                    self.org = self.influxdb_configuration.get('org')
                else:
                    self.config_error = True
                    logger.error(
                        "[AB][InfluxDBConnector] - Influxdb parameters for configuration are not defined in global config file. Check it again!")
            except Exception as e:
                logger.error(
                    f"[AB][InfluxDBConnector] - An exception occurred when trying to load the InfluxDB configurarion. Exception: {e}")

    @logger.catch()
    def connect(self, flag_disconnection):
        """Method that creates the InfluxDB connection
        """
        flag_connection = False
        if self.global_config:
            try:
                url = "http://" + self.host + ":" + str(self.port)
                print(self.org, self.token)
                self.influx_client = InfluxDBClient(
                    url=url, org=self.org, token=self.token, timeout=(6000, 5000))
                self.am_bucket_api = BucketsApi(self.influx_client)
                self.influx_client_writer = self.influx_client.write_api(
                    write_options=WriteOptions(batch_size=1000, flush_interval=10_000))
                status, msg = self.health_and_ping()
                if status:
                    flag_connection = True
                    if flag_disconnection and flag_connection:
                        logger.info(
                            "[AB][InfluxDBConnector] - InfluxDB Connection has been restablished!")
                    logger.warning(
                        f"[AB][InfluxDBConnector] - InfluxDB connection created successfully, but InfluxDB client informs '{msg}' where the status is '{status}'")
                else:
                    logger.error(
                        f"[AB][InfluxDBConnector] - InfluxDB connection has not been created!. InfluxDB client informs '{msg}' and its status is '{status}'")
                    if flag_disconnection and not flag_connection:
                        logger.error(
                            "[AB][InfluxDBConnector] - Trying to reconnect with InfluxDB")
            except Exception as err:
                logger.error(
                    f"[AB][InfluxDBConnector] - An exception has arisen while creating the connection. Exception: {err}")

        return flag_connection, self.influx_client

    @logger.catch()
    def close(self) -> None:
        """Method to close InfluxDB connection
        """
        if self.influx_client:
            self.influx_client.close()
            self.influx_client = None
    
    @logger.catch()
    def health_and_ping(self) -> tuple:
        """Make successive calls until achieve success or a maximum number of attempts.

        Returns:
            Tuple: Status connection and msg
        """
        status_ping = False
        msg = ""
        attempts = 0
        while not status_ping and attempts < 2:
            health = self.influx_client.health()
            status = health.status
            msg = health.message
            if status != 'fail' and self.influx_client.ping():
                status_ping = True
                self.status_ok_connection = True
            attempts += 1
            time.sleep(0.3)
            logger.info("[AB][InfluxDBConnector] - Trying a new ping to the InfluxDB server")
        return status, msg

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
                #    f"[AB][InfluxDBConnector] - Pointname content ({point}) has been inserted in '{bucket}' sucessfully")
            except InfluxDBError as err:
                self.exception_handler(err)
            except Exception as err:
                logger.error(
                    f"[AB][InfluxDBConnector] - An exception has arisen while trying to write the data. Exception: {err}")
        else:
            logger.error(
                "[AB][InfluxDBConnector] - There is no writer defined and also a problem with InfluxDB connection")
    

    @logger.catch()
    def insert_dataframe(self, bucket: str, dataframe: pd.DataFrame, measurement: str, floor: str, other_tags: dict = None):
        if self.influx_client_writer:
            for index, row in dataframe.iterrows():
                point = (
                        Point(measurement)
                        .field(row["_field"], row["_value"])
                        .tag("floor", floor)
                        .time(row["_time"]))
                for tag, value in other_tags.items():
                    point.tag(tag, value)
                extra_tags = row.index.difference(["_field", "_value", "_time"])
                for tag in extra_tags:
                    point.tag(tag, row[tag])
                self.influx_client_writer.write(bucket=bucket, record=point)



    @logger.catch()
    def get_bucket_list(self) -> list:
        """Method to get all the available buckets

        Returns:
            list: Buckets list
        """
        bucket_list = list()
        #if self.influx_client.ping():
        try:
            available_buckets = self.am_bucket_api.find_buckets(limit=100)
            for bucket in available_buckets.buckets:
                bucket_name = bucket.name
                if not bucket.name.startswith("_"):
                    bucket_list.append(bucket_name)                
        except ApiException as err:
            logger.error(
                f"[AB][InfluxDB][Insertion] - There is an ApiException from InfluxDB trying to get buckets list. ApiException: {err}")
        except Exception as err:
            logger.error(
                f"[AB][InfluxDB][Insertion] - There is an Exception from InfluxDB trying to get buckets list. ApiException: {err}")
        else:
            return bucket_list        

    @logger.catch()
    def bucket_creator(self, bucket_candidate: str):
        """Method to create a bucket

        Args:
            bucket_candidate (str): bucket name to verify its existance
            bucket_list (list): bucket list
        """
        bucket_list = self.get_bucket_list()
        if self.influx_client.ping():
            self.bucket_reviewer(bucket_candidate, bucket_list)
            if not self.existence:
                try:
                    self.am_bucket_api.create_bucket(
                        bucket_name=bucket_candidate)
                    logger.info(
                        f"[AB][InfluxDBConnector] - A new bucket has been created: {bucket_candidate}")
                except ApiException as err:
                    logger.error(
                        f"[AB][InfluxDB][Insertion] - There is an ApiException from InfluxDB trying to create a bucket. ApiException: {err}")

    @logger.catch()
    def bucket_reviewer(self, bucket_candidate: str, bucket_list: list):
        """Method to check the bucket existence 

        Args:
            bucket_candidate (str): bucket to be verified
            bucket_list (list): bucket list
        """
        if bucket_list is not None:
            if bucket_candidate in bucket_list:
                self.existence = True
            else:
                self.existence = False
        else:
            logger.error(
                "[AB][InfluxDBConnector] - There are not buckets available")

    @logger.catch()
    def exception_handler(self, err: Exception):
        """Method to manage InfluxDB Exceptions 
        in the writing process

        Args:
            err (Exxception): exception that has been arisen
        """
        status = err.response.status
        message = err.message
        err_content = f"Status: {status}, Message: {message}, Exception: {err}"
        if status == 400:
            logger.error(
                f"[AB][InfluxDBConnector] - Malformed data to be inserted. Check the data structure. {err_content}")
        elif status == 401:
            logger.error(
                f"[AB][InfluxDBConnector] - Unauthorized access to InfluxDB. Check database credentials (host, port, token, organization). {err_content}")
        elif status == 404:
            logger.error(
                f"[AB][InfluxDBConnector] - Organization or bucket not found. {err_content}")
        elif status == 503:
            logger.error(
                f"[AB][InfluxDBConnector] - Server is temporarily unavailable to accept writes. {err_content}")
        else:
            logger.error(
                f"[AB][InfluxDBConnector] - There is a InfluxDB Exception. Please Review in detail. {err_content}")

    @logger.catch()
    def query(self, query: str, schema: bool = False, pandas: bool = False, stream: bool = False) -> Union[list, pd.DataFrame]:
        """Make a influx simple query

        Args:
            query (str): Query string
            schema (boolean): Only schema return

        Returns:
            Union[list, pd.DataFrame]: All records in list or in a dataframe
        """
        query_api = self.influx_client.query_api()
        if pandas:
            results = self._query_pandas(query_api, query, stream)
        else:
            results = self._query_list(query_api, query, schema)
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
                value = {
                aux_key: ("true" if aux_value is True else "false" if aux_value is False else aux_value)
                for aux_key, aux_value in value.items()}
                if key == "group":
                    columns = value.get("columns")
                    if columns:
                        columns_dumped = json.dumps(columns)
                        value["columns"] = columns_dumped
                    one_template = self.queries[key].format(**value)
                elif key == "pivot":
                    column_key = value.get("column_key")
                    if column_key:
                        column_key_dumped = json.dumps(column_key)
                        value["column_key"] = column_key_dumped
                    row_key = value.get("row_key")
                    if row_key:
                        row_key_dumped = json.dumps(row_key)
                        value["row_key"] = row_key_dumped
                    one_template = self.queries[key].format(**value)    
                else:
                    one_template = self.queries[key].format(**value)
                query += "\n" + one_template
        return query
    
    @logger.catch()
    def to_unix_time(self, range: dict) -> dict:
        """Convert range datetime to unix time

        Args:
            range (dict): Range data, start and stop

        Returns:
            dict: Range start, stop in unix time
        """
        start_unix = '0'
        stop_unix = '0' 
        start = range.get("start")
        stop = range.get("stop")
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            stop_dt = datetime.strptime(stop, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        except Exception as err:
            logger.error(f"The date format is not correct {err}")
        else:
            start_unix = int(start_dt.timestamp())
            stop_unix = int(stop_dt.timestamp())
        return {"start": start_unix, "stop": stop_unix}