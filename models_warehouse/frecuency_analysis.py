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
 
from utils.logger_config import get_logger
logger = get_logger()
import pandas as pd

from tdigest import TDigest

from metaclass.templates import MetaModel
import datetime



class FrequencyAnalysis(MetaModel):
    def __init__(self, **parameters):
        """
        Initializes the FrequencyAnalysis model.

        Args:
            **parameters: Additional parameters for the model.
        """
        super().__init__()
        self.LIBRARY_VERSION = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.LIBRARY_FIT_DATE = self.LIBRARY_VERSION
    
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
        self.results_dictionary = {
            "frequency_analysis": predict_freqmatrix,}
        return prediction_results

    @logger.catch()
    def get_info(self) -> str:
        """Returns a string containing information about the model.
        This includes the class name, version, and a brief description of its functionality.

        Returns:
            str: A string containing information about the model.
        """
        info = f"{self.__class__.__name__} - Version: {self.LIBRARY_VERSION}\nDescription: This model establishes data quality.\n"
        info += "It calculates the frequency of each value grouped by 'nae', computes TDigest for temperature, humidity, and CO2, and determines outlier ranges based on the TDigest data."
        info += f"\nFit Date: {self.LIBRARY_FIT_DATE}\n"
        return info
    
    @logger.catch()
    def get_results(self) -> dict:
        """Returns the result dictionary.

        Returns:
            dict: A dictionary containing the results of the analysis.
        """
        return self.results_dictionary
    
    @logger.catch()
    def get_all_attributes(self):
        """Returns all attributes of the model.

        Returns:
            dict: A dictionary containing all attributes of the model.
        """
        return {
            "freq_matrix": self.freq_matrix,
            "results_dictionary": self.results_dictionary,
        }
     
    
    @logger.catch()   
    def update_parameters(self, **attributes):
        """
        Update the model's attributes with the given values.

        Args:
            **attributes: A dictionary of attributes to update.
        """
        self.update(**attributes)
    
    @logger.catch()
    def predict_proba(self, X: pd.DataFrame):
        pass
   
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
                actual_frequency_value = 0
                logger.warning(
                    f"Bucket {bucket} or nae {nae} not found in frequency matrix.")
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