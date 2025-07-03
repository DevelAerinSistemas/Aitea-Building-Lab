'''
 # @ Author: Héctor Bermúdez Castro <hector.bermudez@aitea.tech>
 # @ Create Time: 2025-06-18
 # @ Modified by: Héctor Bermúdez Castro
 # @ Modified time: 2025-06-18
 # @ Project: Aitea Building Lab
 # @ Description: Analytic template
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from utils.logger_config import get_logger
from utils.file_utils import get_configuration
from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
import os
import csv
import pandas as pd
import datetime
import random

logger = get_logger()

@logger.catch()
def influx_query_test(query: str) -> object:
    conn = InfluxDBConnector()
    conn.connect()
    r = conn.query(
        query = query,
        pandas = True,
        stream = False
    )
    return r

if __name__ == "__main__":
    query = """from(bucket: "demo")
        |> range(start: 1748736000, stop: 1748995200)
        |> filter(fn: (r) => r._measurement == "measurement_1")
        |> filter(fn: (r) => r._field == "field_12" or  r._field == "field_13")
    """
    logger.success(f"Results:\n{influx_query_test(query)}")
