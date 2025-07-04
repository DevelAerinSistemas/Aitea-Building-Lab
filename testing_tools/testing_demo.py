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

from utils.file_utils import get_configuration, load_json_file
from utils.logger_config import get_logger

from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
from aitea_connectors.connectors.postgresql_connector import PostgreSQLConnector
from pipelines.pipeline_executor import PipelineExecutor

import pandas as pd
import numpy as np
import csv
import datetime
import random


logger = get_logger()

@logger.catch
def generate_demo_data(parameters:dict) -> None :
    """
    Generates a CSV file for InfluxDB batch import.

    Args:
        parameters (dict): Dictionary that expects at least the following fields:
            - path (str): The path and name of the CSV file to create (e.g. "this_folder/this_file.csv")
            - measurement_column (str): Name of the measurement column.
            - timestamp_column (str): Name of the timestamp column.
            - start_time (str): Start time in ISO format (e.g., "2025-06-01T00:00:00Z").
            - end_time (str): End time in ISO format (e.g., "2025-06-08T00:00:00Z").
            - interval_minutes (int): Interval between data points in minutes.
            - schema (dict): Configuration for measurements and their fields, e.g., 
                {
                    "measurement_1": {
                        "field_11": "double",
                        "field_12": "long",
                        "field_13": "string"
                        ...
                    },
                    ...
                }
            - tags (dict): Configuration for tags and their values, e.g., 
                {
                    "tag_1": [
                        "tag_11",
                        "tag_12",
                        "tag_13"
                        ...
                    ],
                    ...
                }

    """

    logger.info(f"Creating testing data for InfluxDB using configuration:\n{parameters}")

    path = parameters.get("path")
    measurement_column = parameters.get("measurement_column")
    timestamp_column = parameters.get("timestamp_column")
    start_time = parameters.get("start_time")
    end_time = parameters.get("end_time")
    interval_minutes = parameters.get("interval_minutes")
    schema = parameters.get("schema")
    tags = parameters.get("tags")

    all_tag_keys = sorted(list(tags.keys()))
    all_field_keys = []
    for m, fields in schema.items():
        for f_key in fields.keys():
            if f_key not in all_field_keys:
                all_field_keys.append(f_key)
    all_field_keys.sort() # Consistent column order

    # Prepare header rows for InfluxDB CSV format
    # #datatype: measurement,tag,tag,...,field,field,...,dateTime:RFC3339
    datatype_row = ["#datatype", "measurement"]
    for tag_key in all_tag_keys:
        datatype_row.append("tag") # Tags are always string-like in line protocol, treated as 'tag' type
    datatype_row.append("dateTime:RFC3339")
    for f_key in all_field_keys:
        field_type_str = ""
        for m_cfg in schema.values():
            if f_key in m_cfg:
                field_type_str = m_cfg[f_key]
                break
        datatype_row.append(field_type_str)

    # #group: false for measurement, true for tags, false for fields, false for time
    group_row = ["#group", "false"] # measurement
    for _ in all_tag_keys:
        group_row.append("true") # tags are grouping keys
    group_row.append("false") # time
    for _ in all_field_keys:
        group_row.append("false") # fields are not grouping keys

    # #default: typically empty, or specific defaults if desired
    default_row = ["#default", ""] # measurement
    for _ in all_tag_keys:
        default_row.append("") # default for tags
    default_row.append("") # default for time
    for f_key in all_field_keys: # default for fields
        default_val_str = ""
        # Could set type-specific defaults here if needed, e.g., 0 for long/double
        # For simplicity, keeping them empty
        default_row.append(default_val_str)

    # Actual column headers for the data
    header_row = ["", measurement_column] + all_tag_keys + [timestamp_column] + all_field_keys

    start_time = datetime.datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end_time = datetime.datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    current_time = start_time
    time_delta = datetime.timedelta(minutes=interval_minutes)

    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write InfluxDB specific headers
        csv_writer.writerow(datatype_row)
        csv_writer.writerow(group_row)
        csv_writer.writerow(default_row)

        # Write the actual column headers
        csv_writer.writerow(header_row)

        logger.info(f"Generating data from '{start_time}' to '{end_time}' with '{interval_minutes}' min interval.")
        count = 0
        while current_time < end_time:
            # Randomly pick a measurement
            chosen_measurement_name = random.choice(list(schema.keys()))
            measurement_fields = schema[chosen_measurement_name]

            # Prepare data row with placeholders for all possible fields
            data_values = {f_key: None for f_key in all_field_keys}
            tag_values = {t_key: random.choice(tags[t_key]) for t_key in all_tag_keys}

            for field_name, field_type in measurement_fields.items():
                if field_type == "double":
                    data_values[field_name] = round(random.uniform(0.0, 1000.0), 3)
                elif field_type == "long":
                    data_values[field_name] = random.randint(0, 10000)
                elif field_type == "string":
                    # Generate a random short string
                    data_values[field_name] = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(5,10)))
                elif field_type == "boolean":
                    data_values[field_name] = random.choice([True, False]) # csv.writer will handle bools as 'True'/'False'

            # Construct the full row for CSV
            row_to_write = ["", chosen_measurement_name]
            for t_key in all_tag_keys:
                row_to_write.append(tag_values[t_key])
            row_to_write.append(current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4]+"Z")
            for f_key in all_field_keys:
                row_to_write.append(data_values[f_key])

            csv_writer.writerow(row_to_write)
            current_time += time_delta
            count += 1
        logger.success(f"Generated {count} data points into {path}")

@logger.catch
def upload_demo_data_to_influx(influxdb_conn: InfluxDBConnector, testing_conf: dict, testing_df: pd.DataFrame) -> None:
    """TBD"""
    influxdb_conn.bucket_creator(testing_conf.get("bucket"))
    measurement_column = testing_conf.get("data").get("measurement_column")
    for val in testing_df[measurement_column].unique():
        influxdb_conn.insert_dataframe(
            bucket = testing_conf.get("bucket"),
            dataframe = testing_df[testing_df[measurement_column]==val].reset_index(drop=True).drop(columns=[measurement_column]), 
            tags = list(testing_conf.get("data").get("tags").keys()),
            measurement = val,
            timestamp = testing_conf.get("data").get("timestamp_column")
        )

@logger.catch
def _map_pandas_to_postgres_type(dtype: np.dtype) -> str:
    """Maps a pandas dtype to a corresponding PostgreSQL data type."""
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    # Default to TEXT for object, string, and other types
    return "TEXT"

@logger.catch
def _format_sql_value(value):
    """Formats a Python value for use in a raw SQL query string."""
    if pd.isna(value):
        return "NULL"
    elif isinstance(value, str):
        # Escape single quotes by doubling them up for SQL
        return f"""'{value.replace("'", "''")}'"""
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    # For numbers, dates, etc., psycopg2 handles them well, but for raw string
    # it's safest to convert to string. Dates should be quoted.
    elif pd.api.types.is_datetime64_any_dtype(type(value)):
        return f"'{value}'"
    else:
        return str(value)

@logger.catch
def upload_demo_data_to_postgresql(
    pg_connector: PostgreSQLConnector,
    testing_df: pd.DataFrame,
    table_name: str,
    chunk_size: int = 500,
):
    """
    Creates a table in PostgreSQL and uploads data from a pandas DataFrame.

    This function uses only the `execute` method of the provided connector.
    It first drops the table if it exists, then creates a new one based on
    the DataFrame's schema, and finally inserts the data in batches.

    Args:
        pg_connector (PostgreSQLConnector): An instantiated connector object.
        testing_df (pd.DataFrame): The DataFrame containing the data to upload.
        table_name (str): The name of the table to create in the database.
        chunk_size (int): The number of rows to insert in a single batch.
    """
    if testing_df.empty:
        logger.warning(f"DataFrame is empty. No action taken for table '{table_name}'.")
        return

    # Ensure table name is a valid SQL identifier (basic sanitization)
    safe_table_name = "".join(c for c in table_name if c.isalnum() or c == '_')
    if safe_table_name != table_name:
        raise ValueError(f"Invalid table name '{table_name}'. Only alphanumeric characters and underscores are allowed.")

    # 1. Drop existing table for a clean slate
    logger.info(f"Dropping table '{safe_table_name}' if it exists...")
    drop_query = f"DROP TABLE IF EXISTS {safe_table_name};"
    pg_connector.execute(drop_query)

    # 2. Create the CREATE TABLE statement
    logger.info(f"Generating CREATE TABLE statement for '{safe_table_name}'...")
    column_definitions = []
    for col_name, dtype in testing_df.dtypes.items():
        pg_type = _map_pandas_to_postgres_type(dtype)
        # Quote column names to handle spaces or special characters
        column_definitions.append(f'"{col_name}" {pg_type}')
    
    create_query = f"CREATE TABLE {safe_table_name} ({', '.join(column_definitions)});"
    pg_connector.execute(create_query)
    logger.success(f"Table '{safe_table_name}' created successfully.")

    # 3. Insert data in batches
    logger.info(f"Preparing to insert {len(testing_df)} rows in chunks of {chunk_size}...")
    
    # Get quoted column names for the INSERT statement
    quoted_cols = [f'"{col}"' for col in testing_df.columns]
    insert_cols_sql = f"({', '.join(quoted_cols)})"
    
    total_rows_inserted = 0
    for i in range(0, len(testing_df), chunk_size):
        chunk_df = testing_df.iloc[i:i + chunk_size]
        
        value_strings = []
        for _, row in chunk_df.iterrows():
            formatted_values = [_format_sql_value(val) for val in row]
            value_strings.append(f"({', '.join(formatted_values)})")
            
        # Construct the multi-row INSERT statement
        values_sql = ",\n".join(value_strings)
        insert_query = f"INSERT INTO {safe_table_name} {insert_cols_sql} VALUES\n{values_sql};"
        
        rows_affected = pg_connector.execute(insert_query)
        total_rows_inserted += rows_affected
        logger.info(f"Inserted batch {i//chunk_size + 1}. Rows affected: {rows_affected}")

    logger.success(f"🚀 Upload complete. Total rows inserted into '{safe_table_name}': {total_rows_inserted}")


if __name__ == "__main__":

    try:
        logger.debug("Aitea Building Lab test started")

        # Configuration
        load_dotenv()
        demo_conf = load_json_file("config/demo.json")
        demo_pipe_plan_path = demo_conf.get("pipe_plan_path")
        demo_pipe_plan = load_json_file(demo_pipe_plan_path).get("demo")

        postgresql_query_params = demo_pipe_plan.get("steps").get("demo.DemoFuse").get("postgresql")
        demo_pipe_plan_postgresql_query = "\n".join(demo_pipe_plan.get("postgresql").get("query_parts")).format(**postgresql_query_params)

        demo_pipe_plan_influx_query = demo_pipe_plan.get("influxdb").get("training_query")
        demo_pipe_plan_influx_query["bucket"] = demo_pipe_plan_influx_query["buckets"][0]
        del demo_pipe_plan_influx_query["buckets"]

        logger.success(f"Configuration loaded successfully from {os.getenv('CONFIG_PATH')}")
        
        # Creating demo data
        path = demo_conf.get("data").get("path")
        generate_demo_data(demo_conf.get("data"))
        demo_df = pd.read_csv(path, skiprows=3).drop(columns=["Unnamed: 0"])
        logger.success(f"Demo data correctly generated. Sneak peek:\n{demo_df.head()}")

        # Uploading demo data to InfluxDB
        influxdb = InfluxDBConnector()
        influxdb.connect()
        upload_demo_data_to_influx(influxdb, demo_conf, demo_df)

        # Uploading demo data to PostgreSQL
        postgresql = PostgreSQLConnector()
        postgresql.connect()
        upload_demo_data_to_postgresql(postgresql, demo_df, demo_conf.get("bucket"))
        
        # Retrieving demo data from InfluxDB
        data_influx = influxdb.request_query(query_dict=demo_pipe_plan_influx_query, pandas=True)
        influxdb.disconnect()
        logger.info(f"Demo data correctly retrieved from Influx database. Sneak peek:\n{data_influx.head()}")

        # Retrieving demo data from PostgreSQL
        data_postgresql = postgresql.query_to_df(demo_pipe_plan_postgresql_query)
        postgresql.disconnect()
        logger.info(f"Demo data correctly retrieved from PostgreSQL database. Sneak peek:\n{data_postgresql.head()}")

        # # Demo pipelines generating model .pkl and library .so
        # pipe = PipelineExecutor(demo_pipe_plan_path, generate_so=True, save_in_joblib=False)
        # pipe.pipes_executor(testing=False)
        # logger.success("Pipeline execution was successful")

    except Exception as err:
        logger.error(f"Error found when running test: {err}")
    else:
        logger.success("Aitea Building Lab test was a success!")
    finally:
        logger.debug("Aitea Building Lab test completed")