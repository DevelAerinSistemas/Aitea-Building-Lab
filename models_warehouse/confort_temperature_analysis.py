'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-09-19 09:55:18
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-09-19 09:55:24
 # @ Proyect: Aitea Building Lab
 # @ Description: Class that establishes the degrees of comfort based on modification of the set point
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''


from utils.logger_config import get_logger
logger = get_logger()
from typing import Any
import pandas as pd
import numpy as np


from metaclass.templates import MetaTransform


class ConfortTemperatureTransform(MetaTransform):
    def __init__(self, values_dictionary: dict, system_id_name: str = "system_id", datetime_name: str = "_time", set_point_name: list = ["setpoint_effective_temperature", "setpoint_temperature"], star_stop_name: str = "general_condition", downtime: int = 900):
        """This transformation separates the dataframe into rising, plateau, and falling room temperature.

        Args:
            values_dictionary (dict): Set of values ​​for the different attributes, For example, value or values for start/stop
            system_id_name (str, optional): System Id name. Defaults to "system_id".
            datetime_name (str, optional): Datetime name. Defaults to "_time".
            set_point_name (str, optional): Set point attribute name. Defaults to "set_point".
            star_stop_condition_name (str, optional): Star or stop attribute name. Defaults to "general_condition".
        """
        self.datetime_name = datetime_name
        self.set_point_name = set_point_name
        self.star_stop_name = star_stop_name
        self.values_dictionary = values_dictionary
        self.system_id_name = system_id_name
        self.downtime = downtime
        self.system_matrix = None
        self.global_values = None
        self.training_range_dates = None

    def fit(self, X: Any | pd.DataFrame, y: Any | pd.DataFrame = None) -> None:
        """Fit, for compatibility reasons
        Args:
            X (Any | pd.DataFrame): X data
            y (Any | pd.DataFrame, optional): Target. Defaults to None.

        Returns:
            _type_: self
        """
        actual_start_time = X["_time"].min()
        actual_end_time = X["_time"].max()
        if self.training_range_dates:
            last_start_time = self.training_range_dates[0]
            last_end_time = self.training_range_dates[1]
            self.training_range_dates[0] = max(actual_start_time, last_start_time)
            self.training_range_dates[1] = max(actual_end_time, last_end_time)
        else:
            self.training_range_dates = [actual_start_time, actual_end_time]
        X_transform = self._calculate_initial_transform(X)
        if X_transform is None:
            raise Exception
        matrix_dictionary = self._calculate_speed(X_transform)
        if self.system_matrix is None:
            self.system_matrix = pd.DataFrame(matrix_dictionary)
            self.system_matrix = self.system_matrix.set_index("system_id")
        else:
            new_matrix = pd.DataFrame(matrix_dictionary)
            new_matrix = new_matrix.set_index("system_id")
            self.system_matrix = pd.concat([self.system_matrix, new_matrix])
            self.system_matrix = self.system_matrix.groupby('system_id').agg(
                {col: 'sum' if col != "floor" or col != "building" else 'first' for col in self.system_matrix.columns if col != 'group'})
        return self

    def transform(self, X: Any | pd.DataFrame) -> pd.DataFrame:
        """Transform a data set. Basically it is left with the intervals where the system is in operation
        Args:
            X (Any | pd.DataFrame): Input dataset

        Returns:
            pd.DataFrame: Transformed dataset
        """
        X_transform = self._calculate_initial_transform(X)
        system_matrix_copy = self.system_matrix.copy()
        sample_matrix = pd.DataFrame(self._calculate_speed(X_transform))
        sample_matrix = sample_matrix.set_index("system_id").copy()
        common_index = system_matrix_copy.index.intersection(
            sample_matrix.index)
        system_matrix_predict = system_matrix_copy.loc[common_index]
        system_matrix_predict.loc[:,
                                  "total_new_time"] = sample_matrix["total_time"]
        system_matrix_predict.loc[:,
                                  "total_new_changes"] = sample_matrix["total_changes"]
        return system_matrix_predict

    def fit_transform(self, X: Any | pd.DataFrame, y: None | Any | pd.DataFrame = None, **fit_params) -> pd.DataFrame:
        """Transform after fit

        Args:
            X (Any | pd.DataFrame): Input dataset
            y (None | Any | pd.DataFrame, optional): Target. Defaults to None.

        Returns:
            pd.DataFrame: Transformed dataset
        """
        self.fit(X)
        return self.transform(X)

    def partial_fit(self, X: pd.DataFrame):
        """Update the fit

        Args:
            X (pd.DataFrame): Data

        Raises:
            Exception: No data 
        """
        
        X_transform = self._calculate_initial_transform(X)
        if X_transform is None:
            raise Exception
        matrix_dictionary = self._calculate_speed(X_transform)
        new_matrix = pd.DataFrame(matrix_dictionary)
        df_merged = pd.merge(new_matrix, self.system_matrix, on='system_id', suffixes=('_df1', '_df2'), how='outer')
        df_merged.loc[:,'total_changes'] = df_merged['total_changes_df1'].fillna(0) + df_merged['total_changes_df2'].fillna(0)
        df_merged.loc[:,'total_time'] = df_merged['total_time_df1'].fillna(0) + df_merged['total_time_df2'].fillna(0)
        df_merged.loc[:, 'total_setpoints'] = (df_merged['total_setpoints_df1'].fillna(0) + df_merged['total_setpoints_df2'].fillna(0)).astype('int')
        df_merged.loc[:,'floor'] = df_merged['floor_df1'].fillna(df_merged['floor_df2'])
        df_merged.loc[:,'equipment_number'] = df_merged['equipment_number_df1'].fillna(df_merged['equipment_number_df2'])
        df_merged.loc[:,'building'] = df_merged['building_df1'].fillna(df_merged['building_df2'])
        df_merged.loc[:, 'setpoint_last'] = np.where(df_merged['setpoint_last_df2'].notna(), df_merged['setpoint_last_df2'], df_merged['setpoint_last_df1'])
        weighted_mean_1 = df_merged['setpoint_mean_df1'].fillna(0) * df_merged['total_setpoints_df1'].fillna(0)
        weighted_mean_2 = df_merged['setpoint_mean_df2'].fillna(0) * df_merged['total_setpoints_df2'].fillna(0)
        total_setpoints = df_merged['total_setpoints_df1'].fillna(0) + df_merged['total_setpoints_df2'].fillna(0)
        df_merged.loc[:, 'setpoint_mean'] = (weighted_mean_1 + weighted_mean_2) / total_setpoints
        df_merged = df_merged[['system_id', 'equipment_number', 'total_changes', 'total_time', 'floor', 'building', 'setpoint_mean', 'setpoint_last', 'total_setpoints']]
        df_merged = df_merged.set_index('system_id')
        self.system_matrix = df_merged
        

    def _calculate_initial_transform(self, X: Any | pd.DataFrame) -> pd.DataFrame:
        """Create the first transformation of the data. It is left with starting moments, and with the differences in instructions.

        Args:
            X (Any | pd.DataFrame): Dataframe values

        Returns:
            pd.DataFrame: Dataframe with new features
        """
        set_point_name = None
        columns_names = X.columns
        for set_att in self.set_point_name:
            if set_att in columns_names:
                set_point_name = set_att
                break
        X_transform = None
        if set_point_name:
            transform_dataframe_list = list()
            star_values = self.values_dictionary.get("start_values")
            for id in X[self.system_id_name].unique():
                data = X[X[self.system_id_name] == id].copy()
                data.loc[:, "start"] = data[self.star_stop_name].apply(
                    lambda x: True if x in star_values else False)
                data_lenght = data.shape
                if data_lenght[0] < 10:
                    logger.error(f" Not enough data to train in {id}")
                    continue
                #data_in_start = data[data["start"]].copy()
                data.loc[:, "diff"] = data[set_point_name].diff(
                )
                data.loc[:, "diff"] = data["diff"].fillna(0)
                transform_dataframe_list.append(data)
            if len(transform_dataframe_list) > 0:
                X_transform = pd.concat(transform_dataframe_list)
        else:
            logger.error("There is no value for the setpoint in data")
        return X_transform

    def _calculate_speed(self, X: Any) -> pd.DataFrame:
        """Calculate the speed, in changes per second. Only use close times (< self.downtime) to ignore shutdown intervals

        Args:
            X (Any | pd.DataFrame): Dataframe transformed by _calculate_initial_transform. 

        Returns:
            pd.DataFrame: Dataframe with new attributes (total_changes, total_time)
        """
        matrix_ditionary = {"system_id": [], "equipment_number": [],
                            "total_changes": [], "total_time": [], "floor": [], "building": [], "setpoint_mean": [], "setpoint_last": [], "total_setpoints": []}
        for id in X[self.system_id_name].unique():
            data = X[X[self.system_id_name] == id].copy()
            floor = data["floor"].unique()[0]
            equipment_number = data["equipment_number"].unique()[0]
            building = data["building"].unique()[0]
            total_changes = data["diff"].abs().sum()
            data.loc[:, "time_diff"] = data[self.datetime_name].diff().shift(-1)
            dataframe_regular = data[data['time_diff'] < self.downtime]
            total_time = dataframe_regular[dataframe_regular["start"]]["time_diff"].sum(
            )/60
            matrix_ditionary["total_setpoints"].append(data.shape[0])
            setpoint_mean = data['setpoint_temperature'].mean()
            matrix_ditionary["setpoint_mean"].append(setpoint_mean)
            matrix_ditionary["setpoint_last"].append(
                data.iloc[-1]["setpoint_temperature"])
            matrix_ditionary["system_id"].append(id)
            matrix_ditionary["equipment_number"].append(equipment_number)
            matrix_ditionary["total_changes"].append(total_changes)
            matrix_ditionary["total_time"].append(total_time)
            matrix_ditionary["floor"].append(floor)
            matrix_ditionary["building"].append(building)
        return matrix_ditionary
    


if __name__ == "__main__":
    import time
    from datetime import datetime, timedelta
    current_time = datetime.now()
    dates = []
    for i in range(0, 50, 5):
        time = current_time + timedelta(minutes=i)
        dates.append(int(time.timestamp()))
    h1 = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
    t1 = [22, 23, 22, 22, 21, 20, 21, 22, 22, 22]
    h2 = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
    t2 = [22, 23, 22, 22, 21, 20, 21, 22, 22, 22]
    h3 = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
    t3 = [22, 23, 22, 22, 21, 20, 21, 22, 22, 22]
    data_frame1 = {"equipment_number": 10*["20"],  "system_id": 10*["v101"], "floor": 10*[1], "_time": dates,
                   "general_condition": h1,  "setpoint_temperature": t1, "building": 10*["tu_casa"]}
    data_frame2 = {"equipment_number": 10*["21"], "system_id": 10*["v102"], "floor": 10*[2], "_time": dates,
                   "general_condition": h2,  "setpoint_temperature": t2, "building": 10*["tu_casa"]}
    data_frame3 = {"equipment_number": 10*["22"], "system_id": 10*["v103"], "floor": 10*[3], "_time": dates,
                   "general_condition": h3,  "setpoint_temperature": t3, "building": 10*["tu_casa"]}
    dataframe = pd.concat([pd.DataFrame(data_frame1), pd.DataFrame(
        data_frame2), pd.DataFrame(data_frame3)])
    confort = ConfortTemperatureTransform({"start_values": [2]})
    confort.fit(dataframe)
    current_time = datetime.now() + timedelta(days=1)
    dates = []
    for i in range(0, 50, 5):
        time = current_time + timedelta(minutes=i)
        dates.append(int(time.timestamp()))
    h1 = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
    t1 = [22, 23, 22, 21, 22, 22, 22, 22, 22, 22]
    #
    data_frame_test = {"equipment_number": 10*["22"], "system_id": 10*["v101"], "floor": 10*[1], "_time": dates,
                       "general_condition": h1,  "setpoint_temperature": t1, "building": 10*["tu_casa"]}
    print(confort.system_matrix)
    print(confort.training_range_dates)
    confort.partial_fit(pd.DataFrame(data_frame_test))
    print(confort.system_matrix)
    print(confort.training_range_dates)
    # print(confort.fit_transform(pd.DataFrame(data_frame_test)))
