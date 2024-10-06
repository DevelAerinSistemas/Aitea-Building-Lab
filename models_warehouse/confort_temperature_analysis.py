'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-09-19 09:55:18
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-09-19 09:55:24
 # @ Proyect: Aitea Building Lab
 # @ Description: Class that establishes the degrees of comfort based on modification of the set point
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''


from loguru import logger
from typing import Any
import pandas as pd
import numpy as np



from metaclass.templates import MetaTransform, MetaModel



class ConfortTemperatureTransform(MetaTransform):
    def __init__(self, values_dictionary: dict, system_id_name: str = "system_id", datetime_name: str = "_time", set_point_name: str = "setpoint", star_stop_name: str = "fan_speed_status", comfort_ratings: dict = {"optimun": 0, "good": 0.017, "bad": 0.033, "worst": 0.05}, downtime: int = 600):
        """This transformation separates the dataframe into rising, plateau, and falling room temperature.

        Args:
            values_dictionary (dict): Set of values ​​for the different attributes, For example, value or values for start/stop
            system_id_name (str, optional): System Id name. Defaults to "system_id".
            datetime_name (str, optional): Datetime name. Defaults to "_time".
            set_point_name (str, optional): Set point attribute name. Defaults to "set_point".
            star_stop_condition_name (str, optional): Star or stop attribute name. Defaults to "general_condition".
            comfort_ratings: (dict, optiona): Criterion to classify comfort. 
        """
        self.datetime_name = datetime_name
        self.set_point_name = set_point_name
        self.star_stop_name = star_stop_name
        self.values_dictionary = values_dictionary
        self.system_id_name = system_id_name
        self.comfort_ratings = comfort_ratings
        self.downtime = downtime
        self.system_matrix = None
        self.global_values = None

    def fit(self, X: Any | pd.DataFrame, y: Any | pd.DataFrame = None) -> None:
        """Fit, for compatibility reasons
        Args:
            X (Any | pd.DataFrame): X data
            y (Any | pd.DataFrame, optional): Target. Defaults to None.

        Returns:
            _type_: self
        """
        X_transform = self._calculate_initial_transform(X)
        matrix_ditionary = self._calculate_speed(X_transform)
        if self.system_matrix is None:
            self.system_matrix = pd.DataFrame(matrix_ditionary)
            self.system_matrix = self.system_matrix.set_index("system_id")
        else:
            new_matrix = pd.DataFrame(matrix_ditionary)
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
        system_matrix_copy.loc[:, "speed"] = system_matrix_copy["total_changes"] / \
            system_matrix_copy["total_time"]
        sample_matrix = pd.DataFrame(self._calculate_speed(X_transform))
        sample_matrix = sample_matrix.set_index("system_id").copy()
        sample_matrix.loc[:, "speed"] = sample_matrix["total_changes"] / \
            sample_matrix["total_time"]
        common_index = system_matrix_copy.index.intersection(
            sample_matrix.index)
        system_matrix_predict = system_matrix_copy.loc[common_index]
        system_matrix_predict.loc[:, "new_speed"] = sample_matrix["speed"]
        system_matrix_predict.loc[:, "diff"] = sample_matrix["speed"] - \
            system_matrix_predict["speed"]
        system_matrix_predict.loc[:, "manual_speed_classification_previous"] = system_matrix_predict["speed"].apply(
            self._speed_to_label)
        system_matrix_predict.loc[:, "manual_speed_classification_new"] = system_matrix_predict["new_speed"].apply(
            self._speed_to_label)
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

    def _calculate_initial_transform(self, X: Any | pd.DataFrame) -> pd.DataFrame:
        """Create the first transformation of the data. It is left with starting moments, and with the differences in instructions.

        Args:
            X (Any | pd.DataFrame): Dataframe values

        Returns:
            pd.DataFrame: Dataframe with new features
        """
        transform_dataframe_list = list()
        star_values = self.values_dictionary.get("start_values")
        for id in X[self.system_id_name].unique():
            data = X[X[self.system_id_name] == id].copy()
            data.loc[:, "start"] = data[self.star_stop_name].apply(
                lambda x: True if x in star_values else False)
            data_in_start = data[data["start"]].copy()
            data_in_start.loc[:, "diff"] = data_in_start[self.set_point_name].diff(
            )
            transform_dataframe_list.append(data_in_start)
        X_transform = pd.concat(transform_dataframe_list)
        return X_transform

    def _calculate_speed(self, X: Any) -> pd.DataFrame:
        """Calculate the speed, in changes per second. Only use close times (< self.downtime) to ignore shutdown intervals

        Args:
            X (Any | pd.DataFrame): Dataframe transformed by _calculate_initial_transform. 

        Returns:
            pd.DataFrame: Dataframe with new attributes (total_changes, total_time)
        """
        matrix_ditionary = {"system_id": [],
                            "total_changes": [], "total_time": [], "floor": [], "building": []}
        for id in X[self.system_id_name].unique():
            data = X[X[self.system_id_name] == id].copy()
            floor = data["floor"].unique()[0]
            building = data["building"].unique()[0]
            total_changes = data["diff"].abs().sum()
            data.loc[:, "time_diff"] = data[self.datetime_name].diff()
            dataframe_regular = data[data['time_diff'] < self.downtime]
            total_time = dataframe_regular["time_diff"].sum(
            )/60
            matrix_ditionary["system_id"].append(id)
            matrix_ditionary["total_changes"].append(total_changes)
            matrix_ditionary["total_time"].append(total_time)
            matrix_ditionary["floor"].append(floor)
            matrix_ditionary["building"].append(building)
        return matrix_ditionary

    def _speed_to_label(self, value: float) -> str:
        """Label speeds based on criteria set in self.comfort_ratings

        Args:
            value (float): Speed 

        Returns:
            str: Class
        """
        return_value = "not evaluated"
        if value > self.comfort_ratings.get("worst"):
            return_value = "worst"
        elif value > self.comfort_ratings.get("bad"):
            return_value = "bad"
        elif value > self.comfort_ratings.get("good"):
            return_value = "good"
        elif value == self.comfort_ratings.get("optimun"):
            return_value = "optimun"
        return return_value


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
    data_frame1 = {"system_id": 10*["v101"], "floor": 10*[1], "_time": dates, "general_stop_march": h1,  "setpoint_temperature": t1, "building": 10*["tu_casa"]}
    data_frame2 = {"system_id": 10*["v102"], "floor": 10*[2], "_time": dates, "general_stop_march": h2,  "setpoint_temperature": t2, "building": 10*["tu_casa"]}
    data_frame3 = {"system_id": 10*["v103"], "floor": 10*[3], "_time": dates, "general_stop_march": h3,  "setpoint_temperature": t3, "building": 10*["tu_casa"]}
    dataframe = pd.concat([pd.DataFrame(data_frame1), pd.DataFrame(data_frame2), pd.DataFrame(data_frame3)])
    confort  = ConfortTemperatureTransform({"start_values": [2]})
    confort.fit(dataframe)
    current_time = datetime.now() + timedelta(days=1)    
    dates = []
    for i in range(0, 50, 5):
        time = current_time + timedelta(minutes=i)
        dates.append(int(time.timestamp()))
    h1 = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
    t1 = [22, 23, 22, 21, 22, 22, 22, 22, 22, 22]
    #
    print(confort.get_params())
    data_frame_test = {"system_id": 10*["v101"], "floor": 10*[1], "_time": dates, "general_stop_march": h1,  "setpoint_temperature": t1, "building": 10*["tu_casa"]}
    print(confort.transform(pd.DataFrame(data_frame_test)))
    #print(confort.fit_transform(pd.DataFrame(data_frame_test)))