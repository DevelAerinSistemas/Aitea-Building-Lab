'''
 # @ Project: AItea-Building-Lab
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-06-11 13:46:51
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-06-11 13:47:53
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 '''


from loguru import logger
import pandas as pd

from tdigest import TDigest

from metaclass.templates import MetaModel
import datetime

LIBRARY_VERSION = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")


class DataQualityAnalysis(MetaModel):
    """
    Data Quality Analysis Model
    This model performs data quality analysis on a given DataFrame.
    """

    def __init__(self, **parameters):
        """"
        Initializes the DataQualityAnalysis model."""
        super().__init__()
        self.freq_matrix = pd.DataFrame()
        self.ranges_matrix = pd.DataFrame()
        self.result_dictionary = {}
        self.tdigest_dict = {}
        self.range_sensitivity = parameters.get("sensitivity_threshold", 0.1)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the model to the data.

        Args:
            X (pd.DataFrame): The input data.
            y (pd.Series, optional): The target variable (not used in this model).

        Returns:
            self: Returns the instance of the model.
        """
        self.freq_matrix = self._calculate_frequency(X.copy())
        self._calculate_tdigets(X.copy())
        self._calculate_outlier_range()
        return self

    def fit_predict(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit the model and predict anomalies.

        Args:
            X (pd.DataFrame): The input data.
            y (pd.Series, optional): The target variable (not used in this model). Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame containing the results of the prediction.
        """
        self.fit(X, y)
        return self.predict(X)

    @logger.catch()
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies in the data.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The DataFrame containing the results of the prediction.
        """
        prediction_results = pd.DataFrame()
        predict_freqmatrix = self._predict_freq(X.copy())
        predict_ranges = self._predict_ranges(X.copy())
        self.result_dictionary = {
            "frequency_analysis": predict_freqmatrix,
            "range_analysis": predict_ranges}
        
        if not predict_freqmatrix.empty and not predict_ranges.empty:

            predict_freqmatrix = predict_freqmatrix.reset_index()

            prediction_results = pd.concat(
                [predict_freqmatrix, predict_ranges], axis=0, ignore_index=True)

        elif not predict_freqmatrix.empty:
            prediction_results = predict_freqmatrix
        else:
            prediction_results = predict_ranges
        return prediction_results

    @logger.catch()
    def predict_and_refit(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit the model and predict anomalies.

        Args:
            X (pd.DataFrame): The input data.
            y (pd.Series, optional): The target variable (not used in this model). Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame containing the results of the prediction.
        """
        self._calculate_tdigets(X.copy())
        return self.predict(X)
    
    @logger.catch()
    def predict_proba(self, X: pd.DataFrame):
        pass

    def get_info(self) -> str:
        """Returns a string containing information about the model.
        This includes the class name, version, and a brief description of its functionality.

        Returns:
            str: A string containing information about the model.
        """
        info = f"{self.__class__.__name__} - Version: {LIBRARY_VERSION}\nDescription: This model establishes data quality.\n"
        info += "It calculates the frequency of each value grouped by 'nae', computes TDigest for temperature, humidity, and CO2, and determines outlier ranges based on the TDigest data."
        return info

    def get_results(self) -> dict:
        """Returns the result dictionary.

        Returns:
            dict: A dictionary containing the results of the analysis.
        """
        return self.result_dictionary

    def get_all_attributes(self):
        """Returns all attributes of the model.

        Returns:
            dict: A dictionary containing all attributes of the model.
        """
        return {
            "ranges_matrix": self.ranges_matrix,
            "freq_matrix": self.freq_matrix,
            "result_dictionary": self.result_dictionary,
            "tdigest_dict": self.tdigest_dict
        }
    
    
    @logger.catch()    
    def update_parameters(self, **attributes):
        """
        Update the model's attributes with the given values.

        Args:
            **attributes: A dictionary of attributes to update.
        """
        for name, value in attributes.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                logger.warning(f"Attribute '{name}' not found in {self.__class__.__name__}.")

    @logger.catch()
    def _predict_freq(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the frequency of each value in the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame containing the data to analyze.
                It should have columns 'bucket', 'nae', 'pointname', and '_time'.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted frequency of each value.
        """
        freq_predictions = pd.DataFrame()
        actual_frequency = self._calculate_frequency(X.copy())
        freq_predictions['frequency_deviation'] = 0
        for index, row in actual_frequency.iterrows():
            nae = index
            bucket = row['bucket']
            if nae in self.freq_matrix.index and bucket in self.freq_matrix['bucket'].values:
                expected_frequency = self.freq_matrix.loc[nae,
                                                          'mean_time_diff']
                actual_frequency_value = row['mean_time_diff']
                deviation = (actual_frequency_value/expected_frequency)
                freq_predictions.loc[index, 'frequency_deviation'] = deviation
            else:
                freq_predictions.loc[index, 'frequency_deviation'] = 0
            freq_predictions.loc[index, 'bucket'] = bucket
            freq_predictions.loc[index, 'actual_frequency'] = 1/actual_frequency_value if actual_frequency_value != 0 else 0
        freq_predictions = freq_predictions.reset_index()
        if self.freq_matrix.empty:
            logger.warning(
                "Frequency matrix is empty. Please fit the model first.")
        else:
            freq_predictions = freq_predictions.rename_axis('nae')
        return freq_predictions

        freq_predictions = self._calculate_frequency(X.copy())
        return freq_predictions

    def _predict_ranges(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict the range of values in the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame containing the data to analyze.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted range of values.
        """
        range_predictions = pd.DataFrame()
        predict_range = list()
        for bucket in X['bucket'].unique():
            X_bucket = X[X.bucket == bucket]
            if X_bucket.empty:
                logger.warning(f"No data found for bucket: {bucket}")
                continue
            temperatures, humidity, co2 = self._calculate_sensors(X_bucket)

            if not temperatures.empty:
                temperatures.loc[:, 'range_status'] = self._calculate_range_status_vectorized(
                    temperatures, 'temperature')
                temperatures.loc[:, 'bucket'] = bucket
                predict_range.append(
                    temperatures[["_value", "_field", "floor", "range_status", "zone", "bucket"]])
            if not humidity.empty:
                humidity.loc[:, 'range_status'] = self._calculate_range_status_vectorized(
                    humidity, 'humidity')
                humidity.loc[:, 'bucket'] = bucket
                predict_range.append(
                    humidity[["_value", "_field", "floor", "range_status", "zone", "bucket"]])
            if not co2.empty:
                co2.loc[:, 'range_status'] = self._calculate_range_status_vectorized(
                    co2, 'co2')
                co2.loc[:, 'bucket'] = bucket
                predict_range.append(
                    co2[["_value", "_field", "floor", "range_status", "zone", "bucket"]])
        if predict_range:
            predict_range = pd.concat(predict_range, ignore_index=True)
            predict_range.columns = [col.lstrip('_') for col in predict_range.columns]
            range_predictions = predict_range

        return range_predictions

    @logger.catch()
    def _calculate_range_status_vectorized(self, X_bucket: pd.DataFrame, sensor_type: str) -> pd.Series:
        """Calculate the range status for each value in the DataFrame using vectorized operations.

        Args:
            X (pd.DataFrame): The input DataFrame containing the data to analyze.
                It should have columns 'bucket', '_value', and 'sensor_type'.

        Returns:
            pd.Series: A Series containing the range status for each value.
        """
        status = pd.Series([])
        if X_bucket.empty:
            logger.warning(
                "Ranges matrix is empty. Please fit the model first.")
        else:
            bucket_row = self.ranges_matrix[self.ranges_matrix['bucket']
                                            == X_bucket['bucket'].iloc[0]]
        if not bucket_row.empty:
            min_value = bucket_row[f"{sensor_type}_min"].values[0] * \
                (1 - self.range_sensitivity)
            max_value = bucket_row[f"{sensor_type}_max"].values[0] * \
                (1 + self.range_sensitivity)

            status = pd.Series(['within range'] *
                               len(X_bucket), index=X_bucket.index)
            status[X_bucket['_value'] < min_value] = 'outlier to bellow'
            status[X_bucket['_value'] > max_value] = 'outlier to above'
        return status

    @logger.catch()
    def _calculate_range_status(self, value: float, bucket: str, sensor_type: str) -> str:
        """Calculate the range status of a value based on the specified bucket and sensor type.

        Args:
            value (float): Sensor value to check.
            bucket (str): Bucket Name
            sensor_type (str): Sensor type (e.g., temperature, humidity, CO2).

        Returns:
            str: Description of the range status.
        """
        return_range_status = "within range"
        if not self.ranges_matrix.empty:
            if bucket in self.ranges_matrix['bucket'].values:
                row = self.ranges_matrix[self.ranges_matrix['bucket'] == bucket]
                if not row.empty:
                    min_value = row[f"{sensor_type}_min"].values[0] * \
                        (1 - self.range_sensitivity)
                    max_value = row[f"{sensor_type}_max"].values[0] * \
                        (1 + self.range_sensitivity)
                    if value < min_value:
                        return_range_status = "outlier to bellow"
                    elif value > max_value:
                        return_range_status = "outlier to above"
                    else:
                        return_range_status = "within range"
        return return_range_status

    @logger.catch()
    def _calculate_frequency(self, X: pd.DataFrame):
        """
        Calculate the frequency of each value groupby nae.

        Args:
        X (pd.DataFrame): The input DataFrame containing the data to analyze.
            It should have columns 'bucket', 'nae', 'pointname', and '_time'.

        Returns:
            pd.DataFrame: A DataFrame containing the frequency of each value grouped by 'nae'.
        """
        frequency_list = list()
        for bucket in X['bucket'].unique():
            X_bucket = X[X.bucket == bucket]
            if X_bucket.empty:
                logger.warning(f"No data found for bucket: {bucket}")
                continue
            X_bucket = X_bucket.dropna(subset=['nae'])
            X_bucket.loc[:, '_time'] = pd.to_datetime(
                X_bucket['_time'], errors='coerce')
            X_bucket.loc[:, 'time_diff'] = X_bucket.groupby('pointname')[
                '_time'].diff()
            X_bucket = X_bucket.dropna(subset=['time_diff'])
            X_bucket.loc[:, 'time_diff_seconds'] = X_bucket['time_diff'].dt.total_seconds(
            )
            grouped_data = X_bucket.groupby('nae').agg(
                mean_time_diff=('time_diff_seconds', 'mean')
            )
            grouped_data.loc[:, "bucket"] = bucket
            frequency_list.append(grouped_data)
        freq_matrix = pd.concat(
            frequency_list) if frequency_list else pd.DataFrame()
        return freq_matrix

    @logger.catch()
    def _calculate_tdigets(self, X: pd.DataFrame) -> None:
        """
        Calculate the TDigest for each 'nae' in the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame containing the data to analyze.
        """
        for bucket in X['bucket'].unique():
            X_bucket = X[X.bucket == bucket]
            if X_bucket.empty:
                logger.warning(f"No data found for bucket: {bucket}")
                continue
            temperatures, humidity, co2 = self._calculate_sensors(X_bucket)
            if self.tdigest_dict:
                bucket_tdigest = self.tdigest_dict.get(bucket, {})
                if bucket_tdigest:
                    if not temperatures.empty and "room_temperature_tdigest" in bucket_tdigest:
                        bucket_tdigest["room_temperature_tdigest"].batch_update(
                            temperatures["_value"].to_numpy())
                    if not humidity.empty and "room_humidity_tdigest" in bucket_tdigest:
                        bucket_tdigest["room_humidity_tdigest"].batch_update(
                            humidity["_value"].to_numpy())
                    if not co2.empty and "room_co2_tdigest" in bucket_tdigest:
                        bucket_tdigest["room_co2_tdigest"].batch_update(
                            co2["_value"].to_numpy())
                else:
                    if not temperatures.empty:
                        t_digest_temperature = TDigest()
                        self.tdigest_dict[bucket]["room_temperature_tdigest"] = t_digest_temperature
                    if not humidity.empty:
                        t_digest_humidity = TDigest()
                        self.tdigest_dict[bucket]["room_humidity_tdigest"] = t_digest_humidity
                    if not co2.empty:
                        t_digest_co2 = TDigest()
                        self.tdigest_dict[bucket]["room_co2_tdigest"] = t_digest_co2
            else:
                self.tdigest_dict[bucket] = {}
                if not temperatures.empty:
                    t_digest_temperature = TDigest()
                    t_digest_temperature.batch_update(
                        temperatures["_value"].to_numpy())
                    self.tdigest_dict[bucket]["room_temperature_tdigest"] = t_digest_temperature
                if not humidity.empty:
                    t_digest_humidity = TDigest()
                    t_digest_humidity.batch_update(
                        humidity["_value"].to_numpy())
                    self.tdigest_dict[bucket]["room_humidity_tdigest"] = t_digest_humidity
                if not co2.empty:
                    t_digest_co2 = TDigest()
                    t_digest_co2.batch_update(co2["_value"].to_numpy())
                    self.tdigest_dict[bucket]["room_co2_tdigest"] = t_digest_co2

    @logger.catch()
    def _calculate_sensors(self, X_bucket: pd.DataFrame) -> list:
        """
        Calculate the sensor readings for temperature, humidity, and CO2.
        This method filters the DataFrame for the specified bucket and extracts the relevant sensor readings.
        Args:
            bucket_X (pd.DataFrame): The input DataFrame containing the data to analyze.

        Returns:
            list: A list containing the temperature, humidity, and CO2 readings.
        """

        temperatures = X_bucket[X_bucket['_field'] == "room_temperature"]
        humidity = X_bucket[X_bucket['_field'] == "room_humidity"]
        co2 = X_bucket[X_bucket['_field'] == "room_co2"]
        temperatures = temperatures.dropna(subset=['_value'])
        humidity = humidity.dropna(subset=['_value'])
        co2 = co2.dropna(subset=['_value'])
        temperatures = temperatures[temperatures["_value"] > 0]
        humidity = humidity[humidity["_value"] > 0]
        co2 = co2[co2["_value"] > 0]
        return temperatures, humidity, co2

    @logger.catch()
    def _calculate_outlier_range(self) -> None:
        """
        Calculate the outlier ranges for temperature, humidity, and CO2 based on the TDigest data.
        """
        ranges_dictionary = dict()
        if not self.tdigest_dict:
            logger.warning(
                "No TDigest data available for outlier range calculation.")
        else:
            temperature_list = list()
            humidity_list = list()
            co2_list = list()
            for bucket, tdigests in self.tdigest_dict.items():
                ranges_dictionary.setdefault("bucket", []).append(bucket)
                if "room_temperature_tdigest" in tdigests:
                    room_temperature_tdigest = tdigests["room_temperature_tdigest"]
                    q_temp = [room_temperature_tdigest.percentile(
                        25), room_temperature_tdigest.percentile(75)]

                    iqr_temp = q_temp[1] - q_temp[0]
                    temp_outliers_range_min = q_temp[0] - 2 * iqr_temp
                    temp_outliers_range_max = q_temp[1] + 2 * iqr_temp

                    ranges_dictionary.setdefault(
                        "temperature_min", []).append(temp_outliers_range_min)
                    ranges_dictionary.setdefault(
                        "temperature_max", []).append(temp_outliers_range_max)
                else:
                    ranges_dictionary.setdefault(
                        "temperature_min", []).append(None)
                    ranges_dictionary.setdefault(
                        "temperature_max", []).append(None)

                if "room_co2_tdigest" in tdigests:

                    room_co2_tdigest = tdigests["room_co2_tdigest"]

                    q_co2 = [room_co2_tdigest.percentile(
                        25), room_co2_tdigest.percentile(75)]
                    iqr_co2 = q_co2[1] - q_co2[0]
                    co2_outliers_range_min = q_co2[0] - 2 * iqr_co2
                    co2_outliers_range_max = q_co2[1] + 2 * iqr_co2

                    ranges_dictionary.setdefault(
                        "co2_min", []).append(co2_outliers_range_min)
                    ranges_dictionary.setdefault(
                        "co2_max", []).append(co2_outliers_range_max)

                if "room_humidity_tdigest" in tdigests:

                    room_humidity_tdigest = tdigests["room_humidity_tdigest"]

                    q_humidity = [room_humidity_tdigest.percentile(
                        25), room_humidity_tdigest.percentile(75)]
                    iqr_humidity = q_humidity[1] - q_humidity[0]
                    humidity_outliers_range_min = q_humidity[0] - \
                        2 * iqr_humidity
                    humidity_outliers_range_max = q_humidity[1] + \
                        2 * iqr_humidity

                    ranges_dictionary.setdefault("humidity_min", []).append(
                        humidity_outliers_range_min)
                    ranges_dictionary.setdefault("humidity_max", []).append(
                        humidity_outliers_range_max)

        self.ranges_matrix = pd.DataFrame(ranges_dictionary)
