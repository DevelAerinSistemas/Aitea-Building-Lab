'''
 # @ Project: AItea-Brain
 # @ Author: Jose Luis Blanco
 # @ Create Time: 2024-10-16 11:01:00
 # @ Description: Model testing
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Jose Luis Blanco
 # @ Modified time: 2024-10-16 11:01:03
 # @ Copyright (c): 2024 - 2024 Aitea Tech S. L. copying, distribution or modification not authorised in writing is prohibited
 '''


from utils.data_utils import synchronization_and_optimization
from database_tools.influxdb import InfluxDBConnector

from loguru import logger 
from execution import executor
import pandas as pd
import pickle



def queries(query_buckets, new_dates):
    influx = InfluxDBConnector()
    _, _ = influx.connect(True)
    buckets = query_buckets.get("bucket", {}).get("bucket")
    dataframe_list = list()
    if isinstance(buckets, list):
        for bucket in buckets:
            query_buckets["bucket"]["bucket"] = bucket
            if len(new_dates) > 0:
                query_buckets["range"]["start"] = new_dates[0]
                query_buckets["range"]["stop"] = new_dates[1]
            query = influx.compose_influx_query_from_dict(query_buckets)
            stream_data = influx.query(
            query=query, pandas=True, stream=True)
            dataframe = synchronization_and_optimization(stream_data)
            if dataframe is None:
                continue
            if dataframe.empty:
                logger.error(f"Empty search, nothing to do for this bucket {bucket}.")
            dataframe.loc[:, "building"] = bucket
            dataframe_list.append(dataframe)
    if len(dataframe_list) > 0:
        total_dataframe = pd.concat(dataframe_list) 
    return total_dataframe

def testing_pipe():
    model = executor.PipeExecutor("training_models/temperature_confort.pkl")
    pipe = model.pipe.get("pipe")
    query_buckets = model.get_query()
    system_matrix = pipe[0].system_matrix["building"].unique()
    print(system_matrix)


def confort_testing():
    model = executor.PipeExecutor("training_models/temperature_confort.pkl")
    query_buckets = model.get_query()
    pipe = model.pipe.get("pipe")
    system_matrix = pipe[0].system_matrix
    dataframe = queries(query_buckets, ["2024-09-15T12:00:00.000Z", "2024-09-15T16:00:00.000Z"])  
    trans = model.run_transform(dataframe)
    trans.to_pickle("test/testing.pkl")
    system_matrix.to_pickle("test/system.pkl")


    
if __name__ == "__main__":
    confort_testing()
