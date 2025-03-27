'''
 # @ Project: AItea-Building-Lab
 # @ Author: Jose Luis Blanco
 # @ Create Time: 2025-03-17 12:23:50
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Jose Luis Blanco
 # @ Modified time: 2025-03-19 11:53:07
 # @ Copyright (c): 2024 - 2024 Aitea Tech S. L. copying, distribution or modification not authorised in writing is prohibited
 '''


from loguru import logger
from typing import Any
import pandas as pd
import time

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


from metaclass.templates import MetaModel


class ConsumptionAnalysis(MetaModel):
    """Consumption Analysis
    """

    def __init__(self, columns: list, model_type: str = "IF", fit_attributes: list = ["_value", "is_weekend", "hour"], freq: float = 0.5, z_score_threshold: float = 2.4) -> None:
        """Constructor

        Args:
            columns (list): Columns to be used in the analysis
            target (str): Target column
        """
        self.models_matrix = {}
        self.columns = columns
        if "bucket" not in self.columns:
            self.columns.append("bucket")
        self.model_type = model_type
        self.fit_attributes = fit_attributes
        self.freq = freq
        self.z_score_threshold = z_score_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Fit the model

        Args:
            X (pd.DataFrame): Data
            y (pd.Series, optional): Target. Defaults to None.
        """
        timi_i = time.time()
        fitected_bucket = list()
        logger.info(f"[{self.__class__.__name__}] Fitting the model")
        try:
            X_copy = X[self.columns].copy()
        except KeyError as e:
            logger.error(
                f"[{self.__class__.__name__}] - Column not found: {e}, the fit is not possible")
        else:
            X_copy.loc[:, "_time"] = pd.to_datetime(X_copy["_time"])
            X_copy = X_copy.sort_values("_time")
            X_copy = X_copy.dropna()
            buildings = X_copy["bucket"].unique()
            for b in buildings:
                building_X = X_copy[X_copy["bucket"] == b]
                self.models_matrix[b] = self._fit_for_building(building_X)
                fitected_bucket.append(b)
            logger.info(
                f"[{self.__class__.__name__}] Model fitted in {time.time()-timi_i} seconds. Total fitted buckets: {fitected_bucket}")
        return self

    def fit_predict(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Fit and predict

        Args:
            X (pd.DataFrame): Data to fit
            y (pd.Dataframe, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.fit(X, y)
        return self.predict(X)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict

        Args:
            X (pd.DataFrame): Data
        Returns:
            dict: Prediction dictionary
        """
        logger.info(f"[{self.__class__}] Start predicting")
        X = X.reset_index(drop=False)
        X.loc[:, "_time"] = pd.to_datetime(X["_time"])
        X = X.dropna()
        return self._transform_and_predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def get_results(self, prediction: pd.DataFrame) -> dict:
        """Get the results

        Args:
            predict (pd.DataFrame): Prediction

        Returns:
            dict: Results
        """
        return self._analyzer(prediction)

    def upadate_default_values(self, **args) -> None:
        """Update default values
        """
        logger.info(
            f"[{self.__class__.__name__}] Updating default values {args}")
        for key, value in args.items():
            if key == "fit_attributes":
                self.fit_attributes = value
            elif key == "freq":
                self.freq = value
            elif key == "z_score_threshold":
                self.z_score_threshold = value
            elif key == "model_type":
                self.model_type = value

    def get_matrix(self) -> dict:
        """Get trained matrix in a bucket dictionary

        Returns:
            dict: Matrix bucket dictionary
        """
        return self.models_matrix

    def _analyzer(self, prediction_data: pd.DataFrame) -> tuple:
        """Analyzer the prediction data

        Args:
            prediction_data (pd.DataFrame): Predition data

        Returns:
            tuple: Analysis results
        """
        total_summary = list()
        total_summary_dataframe = None
        floors = prediction_data['floor'].unique()
        for floor in floors:
            one_floor = prediction_data[prediction_data['floor'] == floor]
            modules = one_floor['module'].unique()
            for module in modules:
                one_module = one_floor[one_floor['module'] == module]
                outlier = one_module[one_module['prediction'] == -1]
                outlier = outlier[outlier['z-score'] > self.z_score_threshold]
                outlier_summary = outlier.groupby('activity_period').agg(
                    z_score_mean=('z-score', 'mean'),
                    _value_mean=('_value', 'mean'),
                    is_weekend=('is_weekend', lambda x: int(round(x.mean()))),
                    outlier_hours=('_time', lambda x: x.count()*self.freq)
                )
                if not outlier_summary.empty:
                    total_summary.append(outlier_summary.reset_index().assign(
                        floor=floor, module=module))
        if len(total_summary) > 0:
            total_summary_dataframe = pd.concat(total_summary)
            summary_floor_dataframe = total_summary_dataframe.groupby(['floor']).agg(
                z_score_mean=('z_score_mean', 'mean'),
                _value_mean=('_value_mean', 'mean'),
                is_weekend=('is_weekend', lambda x: int(round(x.mean()))),
                outlier_hours=('outlier_hours', 'sum')
            )
            summary_building = {"_value": summary_floor_dataframe["_value_mean"].mean(
            ), "z_score": summary_floor_dataframe["z_score_mean"].mean()}
            summary_building_dataframe = pd.DataFrame(
                summary_building, index=[0])

        return {"total_consumption_summary": total_summary_dataframe, "building_consumption_summary": summary_building_dataframe}

    def _fit_for_building(self, building_X: pd.DataFrame) -> dict:
        """Transform the data for a bucket

        Args:
            building_X (pd.DataFrame): Data for one bucket

        Returns:
            dict: dictionary with the results  
        """
        dictionary_results_one_building = dict()
        floor = building_X["floor"].unique()
        for f in floor:
            data_floor = building_X[building_X["floor"] == f]
            modules = data_floor["module"].unique()
            result_by_modules = dict()
            for m in modules:
                data_floor_module = data_floor[data_floor["module"] == m]
                data_floor_module_grouped = data_floor_module.groupby(
                    ["_time", "module"])["_value"].sum().reset_index()
                data_floor_module_grouped.loc[:, 'is_weekend'] = data_floor_module_grouped['_time'].dt.dayofweek.apply(
                    lambda x: 1 if x >= 5 else 0)
                data_floor_module_grouped.loc[:,
                                              'hour'] = data_floor_module_grouped['_time'].dt.hour
                data_floor_module_grouped.loc[:, 'module_numeric'] = pd.factorize(
                    data_floor_module_grouped['module'])[0]
                data_floor_module_grouped.loc[:, "activity_period"] = data_floor_module_grouped["hour"].apply(
                    lambda x: "activity" if 6 <= x < 21 else "no_activity")
                data_floor_module_grouped_weekend = data_floor_module_grouped[
                    data_floor_module_grouped["is_weekend"] == 1]
                data_floor_module_grouped_week = data_floor_module_grouped[
                    data_floor_module_grouped["is_weekend"] == 0]
                mean_values_weekend = data_floor_module_grouped_weekend.groupby("activity_period")[
                    "_value"].mean()
                mean_values_week = data_floor_module_grouped_week.groupby("activity_period")[
                    "_value"].mean()
                std_values_weekend = data_floor_module_grouped_weekend.groupby("activity_period")[
                    "_value"].std()
                std_values_week = data_floor_module_grouped_week.groupby("activity_period")[
                    "_value"].std()

                fit_model = self._model(data_floor_module_grouped)
                result_by_modules[m] = {"fit_model": fit_model, "mean_values_weekend": mean_values_weekend,
                                        "mean_values_week": mean_values_week, "std_values_weekend": std_values_weekend, "std_values_week": std_values_week}
            dictionary_results_one_building[f] = result_by_modules
        return dictionary_results_one_building

    def _transform_and_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data to fit the model

        Args:
            X (pd.DataFrame): Data

        Returns:
            pd.DataFrame: Transformed data
        """
        prediction_dictionary = {"bucket": [],
                                 "floor": [], "module": [], "prediction": []}
        columns = [col for col in self.columns if col !=
                   "bucket" and col != "floor"] + ["_time"]
        building_to_predict = X["bucket"].unique()
        floor_to_predict = X["floor"].unique()
        return_value = None
        if self.models_matrix:
            prediction_result_append = []
            for b in building_to_predict:
                for f in floor_to_predict:
                    data_floor = X[X["floor"] == f]
                    modules_to_predict = data_floor["module"].unique()
                    for m in modules_to_predict:
                        try:
                            matrix_fit_data = self.models_matrix[b][f][m]
                        except KeyError:
                            logger.error(
                                f"[{self.__class__.__name__}] - Model not found for bucket: {b}, floor: {f}, module: {m}")
                            continue
                        model = matrix_fit_data["fit_model"]
                        data_floor_module = data_floor[data_floor["module"] == m]
                        if data_floor_module.empty:
                            continue
                        data_floor_module_grouped = data_floor_module.groupby(
                            ["_time", "module"])["_value"].sum().reset_index()
                        data_floor_module_grouped.loc[:, 'is_weekend'] = data_floor_module_grouped['_time'].dt.dayofweek.apply(
                            lambda x: 1 if x >= 5 else 0)
                        data_floor_module_grouped.loc[:,
                                                      'hour'] = data_floor_module_grouped['_time'].dt.hour
                        data_floor_module_grouped.loc[:, 'module_numeric'] = pd.factorize(
                            data_floor_module_grouped['module'])[0]
                        data_floor_module_grouped.loc[:, "activity_period"] = data_floor_module_grouped["hour"].apply(
                            lambda x: "activity" if 6 <= x < 21 else "no_activity")
                        prediction = model.predict(
                            data_floor_module_grouped[self.fit_attributes])
                        data_floor_module_grouped.loc[:,
                                                      "prediction"] = prediction
                        data_floor_module_grouped.loc[:, "floor"] = f
                        data_floor_module_grouped.loc[:, "bucket"] = b
                        data_floor_module_grouped.loc[:, "z-score"] = data_floor_module_grouped.apply(
                            lambda row: self._z_score(row, matrix_fit_data), axis=1)
                        prediction_result_append.append(
                            data_floor_module_grouped)
        else:
            logger.error(
                f"[{self.__class__.__name__}] - Model not fitted, the prediction is not possible")
        if len(prediction_result_append) > 0:
            return_value = pd.concat(prediction_result_append)
        else:
            logger.error(
                f"[{self.__class__.__name__}] - Prediction not possible. The fit data for {building_to_predict} is empty")
        return return_value

    def _model(self, data_models: pd.DataFrame) -> Any:
        """Fit a outlier model

        Args:
            data_models (pd.DataFrame): Data
        Returns:
            Any: Fit model
        """
        if self.model_type == "LOF":
            fit_model = LocalOutlierFactor(
                novelty=True).fit(data_models[self.fit_attributes])
        elif self.model_type == "OCSVM":
            fit_model = OneClassSVM(nu=0.25, gamma=0.35).fit(
                data_models[self.fit_attributes])
        elif self.model_type == "EE":
            fit_model = EllipticEnvelope(
                contamination=0.1).fit(data_models[self.fit_attributes])
        elif self.model_type == "IF":
            fit_model = IsolationForest(
                contamination=0.1).fit(data_models[self.fit_attributes])
        else:
            logger.error(
                f"[{self.__class__.__name__}] - Model not found: {self.model_type}")
        return fit_model

    def _z_score(self, x: float, matrix_fit_values: dict) -> float:
        """Z-score

        Args:
            x (float): data row
            matrix_fit_values (dict): Fit values

        Returns:
            float: Z-score
        """
        is_weekend = x["is_weekend"]
        hour = x["hour"]
        activity_period = x["activity_period"]
        if is_weekend:
            mean = matrix_fit_values["mean_values_weekend"][activity_period]
            std = matrix_fit_values["std_values_weekend"][activity_period]
        else:
            mean = matrix_fit_values["mean_values_week"][activity_period]
            std = matrix_fit_values["std_values_week"][activity_period]
        return (x["_value"] - mean) / std


if __name__ == "__main__":
    from database_tools.influxdb_connector import InfluxDBConnector
    from utils.file_utils import load_json_file
    from dotenv import load_dotenv
    import os
    import pickle

    data_conection = load_json_file(os.getenv("INFLUX_CONNECTION"))
#
    # query_dict = dict()
    # query_dict["bucket"] = {"bucket": "recoletos_37"}
    # query_dict["range"] = {
    #    "start": "2024-10-01T00:11:52.211Z", "stop": "2024-12-10T00:11:52.211Z"}
    # query_dict["tag_is"] = [{"tag_name": "element",
    #                         "tag_value": "electrical_network_analyzer"}]
    # query_dict["filter_field"] = [{"field": "active_energy"}]
    # query_dict["window_aggregation"] = {
    #    "every": "30m", "function": "max", "create_empty": "true"}
    # query_dict["fill"] = {"columns": "_value", "previous": "true"}
    # query_dict["difference"] = {"non_negative": "true"}
#
    # influx = InfluxDBConnector(data_conection)
    # influx.load_configuration()
    # influx.connect(True)
    # query_flux = influx.compose_influx_query_from_dict(query_dict)
    # data = influx.query(query=query_flux, pandas=True)
    columns = ["_time", "bucket", "floor", "module", "_value"]
    # data.loc[:, "bucket"] = "recoletos_37"
    c_anal = ConsumptionAnalysis(columns)
    # c_anal.fit(data)

    # c_anal.upadate_default_values(z_score_threshold=17)

    influx = InfluxDBConnector(data_conection)
    influx.load_configuration()
    influx.connect(True)
    query_dict = dict()
    query_dict["bucket"] = {"bucket": "recoletos_37"}
    query_dict["range"] = {
        "start": "2024-12-10T00:11:52.211Z", "stop": "2024-12-20T10:11:52.211Z"}
    query_dict["tag_is"] = [{"tag_name": "element",
                             "tag_value": "electrical_network_analyzer"}]
    query_dict["filter_field"] = [{"field": "active_energy"}]
    query_dict["window_aggregation"] = {
        "every": "30m", "function": "max", "create_empty": "true"}
    query_dict["fill"] = {"columns": "_value", "previous": "true"}
    query_dict["difference"] = {"non_negative": "true"}
    query_flux = influx.compose_influx_query_from_dict(query_dict)
    data = influx.query(query=query_flux, pandas=True)
    data.loc[:, "bucket"] = "recoletos_36"
    result = c_anal.predict(data)

    print(c_anal.get_results(result))
    # with open("consumption_analysis_model.pkl", "wb") as f:
    #    pickle.dump(c_anal.models_matrix, f)
