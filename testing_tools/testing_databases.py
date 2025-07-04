'''
 # @ Author: Héctor Bermúdez Castro <hector.bermudez@aitea.tech>
 # @ Create Time: 2025-06-18
 # @ Modified by: Héctor Bermúdez Castro
 # @ Modified time: 2025-06-18
 # @ Project: Aitea Building Lab
 # @ Description: InfluxDB test
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from utils.logger_config import get_logger
from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
from aitea_connectors.connectors.postgresql_connector import PostgreSQLConnector

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

@logger.catch()
def postgresql_query_test(query: str) -> object:
    conn = PostgreSQLConnector()
    conn.connect()
    r = conn.query_to_df(query)
    return r

if __name__ == "__main__":
    influx_query = """from(bucket: "demo")
        |> range(start: 1748736000, stop: 1748995200)
        |> filter(fn: (r) => r._measurement == "measurement_1")
        |> filter(fn: (r) => r._field == "field_12" or  r._field == "field_13")
    """
    logger.success(f"Results:\n{influx_query_test(influx_query)}")

    postgresql_query = """SELECT * FROM demo LIMIT 100"""
    logger.success(f"Results:\n{postgresql_query_test(postgresql_query)}")
