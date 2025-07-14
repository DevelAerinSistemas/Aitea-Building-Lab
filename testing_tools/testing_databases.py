'''
 # @ Project: AItea-Brain-Lite
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-06-20
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-06-20
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
from utils.logger_config import get_logger
logger = get_logger()

try:
    from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
    from aitea_connectors.connectors.postgresql_connector import PostgreSQLConnector
    AITEA_CONNECTORS = True
except ImportError:
    logger.warning(f"⚠️ Aitea Connectors are not available. Uncomplete functionality: only local files as a valid data source.")
    AITEA_CONNECTORS = False

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
    if AITEA_CONNECTORS:
        influx_query = """from(bucket: "demo")
            |> range(start: 1748736000, stop: 1748995200)
            |> filter(fn: (r) => r._measurement == "measurement_1")
            |> filter(fn: (r) => r._field == "field_12" or  r._field == "field_13")
        """
        logger.success(f"✅ InfluxDB results:\n{influx_query_test(influx_query)}")

        postgresql_query = """SELECT * FROM demo LIMIT 100"""
        logger.success(f"✅ PostgreSQL results:\n{postgresql_query_test(postgresql_query)}")
    else:
        logger.warning(f"⚠️ No AITEA_CONNECTORS available to test")
