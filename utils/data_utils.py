'''
 # @ Project: AItea-Brain-Lite
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-02-20
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-06-21
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

from utils.logger_config import get_logger
logger = get_logger()
import pandas as pd
import time
import json

def synchronization_and_optimization(stream_data: pd.DataFrame) -> pd.DataFrame:
    """Synchronize and reduce the size of the dataframe received from influx

    Args:
        stream_data (pd.DataFrame (stream)): Stream dataframe (generator)  

    Returns:
        pd.DataFrame: Total dataframe (None if is empty)
    """
    dataframe_list = []
    for one_data in stream_data:
        one_data["_value"] = one_data["_value"].astype('float32') 
        df_selected = one_data.reset_index(
        )[['_time', 'system_id', 'floor', '_field', '_value', 'equipment_number']]
        df_pivot = df_selected.pivot_table(
        # Las columnas que mantendremos como índice
        index=['_time', 'system_id', 'floor', 'equipment_number'],
        # La columna cuyos valores se convertirán en nuevas columnas
        columns='_field',
        values='_value',                        # Los valores de las nuevas columnas
        # Función para manejar duplicados (puedes ajustar si es necesario)
        ).reset_index()
        dataframe_list.append(df_pivot)
    if len(dataframe_list) > 0:
        data_pivot_all = pd.concat(dataframe_list, ignore_index=True)
        data_pivot_all.loc[:, "_time"] = data_pivot_all['_time'].dt.ceil('min')
        data_pivot_all.loc[:, "_time"] = data_pivot_all['_time'].dt.round('5min')
        data_pivot_all = data_pivot_all.groupby(
            ['_time', 'system_id', 'floor', 'equipment_number']).mean().reset_index()
        total_nan =  data_pivot_all.isna().sum().sum()
        logger.info(f"Total number of NaN's after synchronization: {total_nan} in {data_pivot_all.shape}")
        data_pivot_all = fill_nan(data_pivot_all)
        total_nan =  data_pivot_all.isna().sum().sum()
        logger.info(f"Total number of NaN's after synchronization: {total_nan} in {data_pivot_all.shape}")
        data_pivot_all = data_pivot_all.dropna()
        data_pivot_all.loc[:, "_time"] = data_pivot_all['_time'].astype('int64') // 10**9 
        for one_column in data_pivot_all.columns:
            if not one_column in ["_time",]:
                if one_column in ["floor", "system_id", "equipment_number"]:
                    data_pivot_all.loc[:, one_column] = data_pivot_all[one_column].astype('category')
                elif one_column not in ["room_temperature", "setpoint_temperature"]:
                    data_pivot_all.loc[:, one_column] = data_pivot_all[one_column].astype('int8')
                else:
                    data_pivot_all.loc[:, one_column] = data_pivot_all[one_column].astype('float16')
        mem_usage_per_column = data_pivot_all.memory_usage(deep=True)
        mem_usage_total = mem_usage_per_column.sum()
        logger.info(f"Total dataframe size: {mem_usage_total / (1024 ** 2):.2f} MB")
        if data_pivot_all.empty:
            logger.warning("Empty dataframe")
            data_pivot_all = None
    else:
        data_pivot_all = None
        logger.warning("Empty dataframe")
    return data_pivot_all


def fill_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe_list = []
    system_ids = dataframe["system_id"].unique()
    for system_id in system_ids:
        one = dataframe[dataframe["system_id"] == system_id]
        one  = one.ffill(limit=2)
        one  = one.bfill(limit=2)
        dataframe_list.append(one)
    if len(dataframe_list) > 0:
        dataframe = pd.concat(dataframe_list)
    return dataframe



        