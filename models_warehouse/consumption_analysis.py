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


from utils.logger_config import get_logger
logger = get_logger()
from typing import Any
import pandas as pd
import time

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


from metaclass.templates import MetaModel


WEEK_DAYS = [0, 1, 2, 3, 4]
WEEKEND_DAYS = [5, 6]
LABOR_HOURS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
NOT_LABOR_HOURS = [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]


class ConsumptionAnalysis(MetaModel):
    """Consumption Analysis
    """

    def __init__(self, columns: list, model_type: str = "IF", fit_attributes: list = ["_value", "week_day", "hour", "location_numeric"], freq: float = 0.5, z_score_threshold: float = 2.4) -> None:
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
        location_codes, location_categories, building_X_cpy = self._transform(
            building_X, fit=True, building=building_X["bucket"].unique()[0])
        for f in floor:
            data_floor = building_X_cpy[building_X_cpy["floor"] == f]
            modules = data_floor["module"].unique()
            result_by_modules = dict()
            for m in modules:
                data_floor_module = data_floor[data_floor["module"] == m]
                data_floor_module_week = data_floor_module[data_floor_module["week_day"].isin(
                    WEEK_DAYS)]
                data_floor_module_week_labor = data_floor_module_week[data_floor_module_week["hour"].isin(
                    LABOR_HOURS)]
                data_floor_module_week_not_labor = data_floor_module_week[data_floor_module_week["hour"].isin(
                    NOT_LABOR_HOURS)]
                data_floor_module_weekend = data_floor_module[data_floor_module["week_day"].isin(
                    WEEKEND_DAYS)]
                mean_values_weekend = data_floor_module_weekend["_value"].mean(
                )
                std_values_weekend = data_floor_module_weekend["_value"].std()
                mean_values_week_labor = data_floor_module_week_labor["_value"].mean(
                )
                std_values_week_labor = data_floor_module_week_labor["_value"].std(
                )
                mean_values_week_not_labor = data_floor_module_week_not_labor["_value"].mean(
                )
                std_values_week_not_labor = data_floor_module_week_not_labor["_value"].std(
                )
                result_by_modules[m] = {"mean_values_weekend": mean_values_weekend, "std_values_weekend": std_values_weekend, "mean_values_week_labor": mean_values_week_labor,
                                        "std_values_week_labor": std_values_week_labor, "mean_values_week_not_labor": mean_values_week_not_labor, "std_values_week_not_labor": std_values_week_not_labor}
            dictionary_results_one_building[f] = result_by_modules
        dictionary_results_one_building["location_categories"] = location_categories
        model = self._model(building_X_cpy)
        dictionary_results_one_building["model"] = model
        return dictionary_results_one_building

    def _transform(self, X: pd.DataFrame, fit: bool = False, building: str = None) -> pd.DataFrame:
        X_transformed = X.copy()
        X_transformed.loc[:, "location"] = X_transformed["floor"].astype(
            str) + "_" + X["module"].astype(str)
        X_transformed = X_transformed.groupby(["location", "_time"], as_index=False).agg(
            {"_value": "sum", "floor": "first", "module": "first"})
        X_transformed = X_transformed.dropna()
        X_transformed.loc[:, 'week_day'] = X_transformed['_time'].dt.dayofweek
        X_transformed.loc[:, 'hour'] = X_transformed['_time'].dt.hour
        if fit:
            location_codes, location_categories = pd.factorize(
                X_transformed['location'])
            X_transformed.loc[:, 'location_numeric'] = location_codes

        else:
            location_codes = None
            location_categories = self.models_matrix.get(
                building).get("location_categories")
            mapping_dict = {cat: i for i,
                            cat in enumerate(location_categories)}
            X_transformed.loc[:, 'location_numeric'] = X_transformed['location'].map(
                lambda x: mapping_dict.get(x, -1))
        return location_codes, location_categories, X_transformed

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
        return_value = None
        if self.models_matrix:
            prediction_result_append = []
            for b in building_to_predict:
                _, _, data_building_to_predict = self._transform(
                    X[X["bucket"] == b].copy(), fit=False, building=b)
                try:
                    model = self.models_matrix[b].get("model")
                except KeyError:
                    continue
                prediction_result = model.predict(
                    data_building_to_predict[self.fit_attributes])
                data_building_to_predict.loc[:,
                                             "prediction"] = prediction_result
                data_building_to_predict.loc[:, "z-score"] = data_building_to_predict.apply(
                    lambda row: self._z_score(row, self.models_matrix[b][row["floor"]][row["module"]]), axis=1)
                data_building_to_predict.loc[:, "bucket"] = b
                prediction_result_append.append(data_building_to_predict)
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
        fit_model = None
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
        if fit_model:
            self.clean_model(fit_model)
        return fit_model

    def _z_score(self, x: float, matrix_fit_values: dict) -> float:
        """Z-score

        Args:
            x (float): data row
            matrix_fit_values (dict): Fit values

        Returns:
            float: Z-score
        """
        is_weekend = x["week_day"] > 4
        is_labor = x["hour"] in [0, 1, 2, 3,
                                 4, 5, 20, 21, 22, 23] == "activity"
        if is_weekend:
            mean = matrix_fit_values["mean_values_weekend"]
            std = matrix_fit_values["std_values_weekend"]
        else:
            if is_labor:
                mean = matrix_fit_values["mean_values_week_labor"]
                std = matrix_fit_values["std_values_week_labor"]
            else:
                mean = matrix_fit_values["mean_values_week_not_labor"]
                std = matrix_fit_values["std_values_week_not_labor"]
        return (x["_value"] - mean) / std if std != 0 else 0


if __name__ == "__main__":
    from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
    from utils.file_utils import load_json_file
    from dotenv import load_dotenv
    import os
    import pickle

    query_dict = dict()
    query_dict["bucket"] = {"bucket": "recoletos_37"}
    query_dict["range"] = {
        "start": "2025-01-18T00:11:52.211Z", "stop": "2025-01-19T00:11:52.211Z"}
    query_dict["tag_is"] = [{"tag_name": "element",
                            "tag_value": "electrical_network_analyzer"}]
    query_dict["filter_field"] = [{"field": "active_energy"}]
    query_dict["window_aggregation"] = {
        "every": "1h", "function": "max", "create_empty": "true"}
    query_dict["fill"] = {"columns": "_value", "previous": "true"}
    query_dict["difference"] = {"non_negative": "true"}
    influx = InfluxDBConnector()
    influx.connect()
    query_flux = influx.compose_influx_query_from_dict(query_dict)
    data = influx.query(query=query_flux, pandas=True)
    columns = ["_time", "bucket", "floor", "module", "_value"]
    data.loc[:, "bucket"] = "recoletos_37"
    c_anal = ConsumptionAnalysis(columns)
    c_anal.fit(data)

    # c_anal.upadate_default_values(z_score_threshold=17)

    influx = InfluxDBConnector()
    influx.connect()
    query_dict = dict()
    query_dict["bucket"] = {"bucket": "recoletos_37"}
    query_dict["range"] = {
        "start": "2025-01-18T00:11:52.211Z", "stop": "2025-01-19T00:11:52.211Z"}
    query_dict["tag_is"] = [{"tag_name": "element",
                             "tag_value": "electrical_network_analyzer"}]
    query_dict["filter_field"] = [{"field": "active_energy"}]
    query_dict["window_aggregation"] = {
        "every": "1h", "function": "max", "create_empty": "true"}
    query_dict["fill"] = {"columns": "_value", "previous": "true"}
    query_dict["difference"] = {"non_negative": "true"}
    query_flux = influx.compose_influx_query_from_dict(query_dict)
    data = influx.query(query=query_flux, pandas=True)
    data.loc[:, "bucket"] = "recoletos_37"
    result = c_anal.predict(data)

    print(c_anal.get_results(result))
    # with open("consumption_analysis_model.pkl", "wb") as f:
    #    pickle.dump(c_anal.models_matrix, f)
