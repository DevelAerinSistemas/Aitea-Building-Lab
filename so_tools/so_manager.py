'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-10-02 15:46:27
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-10-02 15:46:39
 # @ Proyect: Aitea Building Lab
 # @ Description: Testing so 
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

import pandas as pd
from loguru import logger


from database_tools.influxdb import InfluxDBConnector
from lib import executor
from utils.data_utils import synchronization_and_optimization


class SOManager(object):
    def __init__(self, path: str):
        self.influx = InfluxDBConnector()
        self.flag_connection, _ = self.influx.connect(True)
        logger.info(f"Using version {executor.__version__} of the executor")
        self.exec = executor.PipeExecutor(path=path) 

        
    def generate_dataframe(self, date_ranges: list, query_buckets: dict = None):
        total_dataframe = None
        if not query_buckets:
            query_buckets = self.exec.get_query()
        buckets = query_buckets.get("bucket", {}).get("bucket")
        dataframe_list = list()
        total_dataframe = None
        if self.flag_connection:
            if isinstance(buckets, list):
                for bucket in buckets:
                    query_buckets["bucket"]["bucket"] = bucket
                    query_buckets["range"]["start"] = date_ranges[0]
                    query_buckets["range"]["stop"] = date_ranges[1]
                    stream_data = self.influx.request_query(
                    query_dict=query_buckets, pandas=True)
                    dataframe = synchronization_and_optimization(stream_data)
                    if dataframe.empty:
                        logger.error(f"Empty search, nothing to do for this bucket {bucket}.")
                        continue
                    dataframe.loc[:, "building"] = bucket
                    dataframe_list.append(dataframe) 
                total_dataframe = pd.concat(dataframe_list) 
                if total_dataframe.empty:
                    logger.error(f"Total empty dataframe, nothing to do.")
        else:
            logger.critical("There is no connection to the database, cannot continue.")
        self.influx.close()
        return total_dataframe
    
    def exec_prediction(self, X : pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        return self.exec.run_predict(X, y)

    def exec_transform(self, X : pd.DataFrame) -> pd.DataFrame:
        return self.exec.run_transform(X)

    def exec_get_params(self, position: int = 0) -> dict:
        return self.exec.get_params(position)


    


if __name__ == "__main__":
    exec = SOManager("training_models/temperature_confort.pkl")
    dataframe = exec.generate_dataframe(date_ranges=["2024-09-07T10:00:00.000Z", "2024-09-07T12:00:00.000Z"])



