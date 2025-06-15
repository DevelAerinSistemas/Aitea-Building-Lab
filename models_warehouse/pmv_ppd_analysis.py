'''
 # @ Project: AItea-Bulding-Lab
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-05-23 14:02:58
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-05-23 14:03:27
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 '''


from loguru import logger
import pandas as pd
from pythermalcomfort.models import pmv_ppd_iso

from metaclass.templates import MetaModel
import datetime

LIBRARY_VERSION = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

class PMVPPDAnalysis(MetaModel):
    def __init__(self, thresholds: dict):
        """Constructs a PMVPPDAnalysis model.
        This model is designed to analyze PMV (Predicted Mean Vote) and PPD (Predicted Percentage of Dissatisfied) values based on input data.
        It initializes the model with specified thresholds for various parameters and prepares data structures for analysis.

        Args:
            thresholds (dict): Threshold values for analysis. 
        """
        super().__init__() 
        self.thresholds = thresholds
        self.data_matrix = pd.DataFrame()
        self.bucket_data = pd.DataFrame()
        self.result_dictionary = {} 
    
    def fit(self, X, y=None):
        """Fit the model to the data.
        
        Args:
            X (pd.DataFrame): Input data.
            y (pd.DataFrame, optional): Target data. Defaults to None.
        
        Returns:
            self: Fitted model.
        """
        self.data_matrix, self.bucket_data = self._calculate_data_matrix(X)
        return self

    def fit_predict(self, X, y = None) -> pd.DataFrame:
        """Fit the model and return predictions.
        
        Args:
            X (pd.DataFrame): Input data.
            y (pd.DataFrame, optional): Target data. Defaults to None.
        
        Returns:
            pd.DataFrame: Predictions.
        """
        self.fit(X, y)
        return self.predict(X)
        
    
    def predict(self, X: pd.DataFrame):
        """Predicts PMV and PPD values for the input DataFrame.
        This method processes the input DataFrame to calculate PMV and PPD values for each floor and bucket.

        Args:
            X (pd.DataFrame): Input DataFrame containing the necessary data for prediction.
        """
        try:
            data_prepare_floors, data_prepare_bucket = self._calculate_data_matrix(X)
            if data_prepare_floors.empty or data_prepare_bucket.empty:
                logger.warning("No data found for prediction. Returning empty DataFrames.")
                entire_bucket_prediction = pd.DataFrame()
                floor_buckets_prediction = pd.DataFrame()
            else:
                entire_bucket_prediction = self._predict_entire_bucket(data_prepare_bucket)
                floor_buckets_prediction = self._predict_floor_bucket(data_prepare_floors)
            self.result_dictionary = {
                "building_pmvppd_analysis": entire_bucket_prediction,
                "floor_pmvppd_analysis": floor_buckets_prediction
            }
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}")
            entire_bucket_prediction = pd.DataFrame()
            floor_buckets_prediction = pd.DataFrame()
            self.result_dictionary = {
                "building_pmvppd_analysis": entire_bucket_prediction,
                "floor_pmvppd_analysis": floor_buckets_prediction
            }
        return entire_bucket_prediction

    def get_matrix(self):
        """Returns the data matrix used for predictions.
        This method retrieves the data matrix and bucket data used in the model.

        Returns:
            tuple: A tuple containing the data matrix and bucket data as pandas DataFrames.
        """
        return self.data_matrix, self.bucket_data
    
    def get_results(self):
        """Returns the results of the predictions.
        This method retrieves the result dictionary containing predictions for entire buckets and floor buckets.

        Returns:
            dict: A dictionary containing the predictions for entire buckets and floor buckets.
        """
        return self.result_dictionary
    
    def get_prediction(self, X):
        pass
    
    def predict_proba(self, X):
        pass
    
    def get_all_attributes(self):
        """Returns all attributes of the model.
        
        Returns:
            dict: A dictionary containing all attributes of the model.
        """
        return {
            "data_matrix": self.data_matrix,
            "bucket_data": self.bucket_data,
            "result_dictionary": self.result_dictionary,
            "thresholds": self.thresholds
        }
        
    def get_info(self):
       """Returns information about the model.
       """
       info = f"{self.__class__.__name__} - Version: {LIBRARY_VERSION}\n: This model analyzes PMV and PPD values based on input data.\n"
       info += "It calculates PMV and PPD values for each floor and bucket, providing insights into thermal comfort conditions.\n"
       info += "It uses thresholds for various parameters to filter and analyze the data effectively.\n"
       info += "The model can predict PMV and PPD values for entire buckets and individual floors, allowing for detailed analysis of thermal comfort across different areas.\n"
       info += "It also calculates a system matrix based on the input data, which is used for predictions.\n"
       info += "The predict method provides dataframe with:\n"
       info += " - PMV and PPD values for each bucket, for floor and hour in winter and summer.\n"
       info += " - Humidity, temperature and co2 mean for floor and hour.\n"
       info += " - Difference between the input data and the system matrix for PMV and PPD values.\n"
       info += "The get_results method returns a dictionary containing predictions for entire buckets and floor buckets.\n"
       return info
    
    def _predict_floor_bucket(self, data_prepare_floors: pd.DataFrame) -> pd.DataFrame:
        """Predicts for all floors in the input DataFrame.
        
        Args:
            data_prepare_floors (pd.DataFrame): Input data containing all floor information.
        
        Returns:
            pd.DataFrame: DataFrame with predictions for all floors.
        """
        result = pd.DataFrame()
        for bucket in data_prepare_floors.bucket.unique():
            bucket_data = data_prepare_floors[data_prepare_floors.bucket == bucket]
            if bucket_data.empty:
                logger.warning(f"No data found for bucket {bucket}. Skipping prediction for this bucket.")
                continue
            else:
                data_matrix_bucket = self.data_matrix[self.data_matrix['bucket'] == bucket]
                # Itera sobre cada fila en bucket_data para comparar con self.data_matrix
                hours_list = []
                floor_list = []
                difference_pmv_summer_list = []
                difference_ppd_summer_list = []
                difference_pmv_winter_list = []
                difference_ppd_winter_list = []
                for index, row in bucket_data.iterrows():
                    # Obtiene la hora y la planta de la fila actual
                    hour = row['hour']
                    floor = row['floor']
                    # Filtra data_matrix_bucket para la misma hora y planta
                    comparison_data = data_matrix_bucket[(data_matrix_bucket['hour'] == hour) & (data_matrix_bucket['floor'] == floor)]
                    # Si encuentra datos para comparar
                    if not comparison_data.empty:
                        # Extrae los valores de PMV y PPD de ambos DataFrames
                        pmv_summer_bucket = row['pmv_summer']
                        ppd_summer_bucket = row['ppd_summer']
                        pmv_winter_bucket = row['pmv_winter']
                        ppd_winter_bucket = row['ppd_winter']
                        
                        # Extrae los valores de PMV y PPD de comparison_data
                        pmv_summer_matrix = comparison_data['pmv_summer'].values[0]
                        ppd_summer_matrix = comparison_data['ppd_summer'].values[0]
                        pmv_winter_matrix = comparison_data['pmv_winter'].values[0]
                        ppd_winter_matrix = comparison_data['ppd_winter'].values[0]
                        
                        difference_pmv_summer = pmv_summer_bucket - pmv_summer_matrix
                        difference_ppd_summer = ppd_summer_bucket - ppd_summer_matrix
                        difference_pmv_winter = pmv_winter_bucket - pmv_winter_matrix
                        difference_ppd_winter = ppd_winter_bucket - ppd_winter_matrix
                        
                        # Agrega los resultados a las listas
                        hours_list.append(hour)
                        floor_list.append(floor)
                        difference_pmv_summer_list.append(difference_pmv_summer)
                        difference_ppd_summer_list.append(difference_ppd_summer)
                        difference_pmv_winter_list.append(difference_pmv_winter)
                        difference_ppd_winter_list.append(difference_ppd_winter)
                # Crea un DataFrame con los resultados
                if hours_list:
                    results_df = pd.DataFrame({
                        'hour': hours_list,
                        'floor': floor_list,
                        'difference_pmv_summer': difference_pmv_summer,
                        'difference_ppd_summer': difference_ppd_summer,
                        'difference_pmv_winter': difference_pmv_winter,
                        'difference_ppd_winter': difference_ppd_winter})
                    results_df = bucket_data.merge(results_df, on=['hour', 'floor'], how='left')
            result = pd.concat([result, results_df], ignore_index=True)
        if result.empty:
            result = data_prepare_floors
        return result
                        
    
    def _predict_entire_bucket(self, data_prepare_bucket: pd.DataFrame) -> pd.DataFrame:
        """Predicts for all buckets in the input DataFrame.
        
        Args:
            data_prepare_bucket (pd.DataFrame): Input data containing all bucket information.
        
        Returns:
            pd.DataFrame: DataFrame with predictions for all buckets.
        """
        results = pd.DataFrame()
        if not data_prepare_bucket.empty:
            for bucket in data_prepare_bucket.bucket.unique():
                bucket_data = data_prepare_bucket[data_prepare_bucket.bucket == bucket]
                if not bucket_data.empty:
                    one_bucket_data = self.bucket_data[self.bucket_data['bucket'] == bucket]                
                    hours_list = []
                    difference_pmv_summer_list = []
                    difference_ppd_summer_list = []
                    difference_pmv_winter_list = []
                    difference_ppd_winter_list = []
                    for index, row in bucket_data.iterrows():

                        # Hora de la fila actual
                        hour = row['hour']

                        # Datos para comparar de la matriz de datos del edificio 
                        comparison_data = one_bucket_data[(one_bucket_data['hour'] == hour)]
                        # If data is found for comparison
                        if not comparison_data.empty:

                            # Datos de la fila actual para valores de entrada 
                            pmv_summer_bucket = row['pmv_summer']
                            ppd_summer_bucket = row['ppd_summer']
                            pmv_winter_bucket = row['pmv_winter']
                            ppd_winter_bucket = row['ppd_winter']

                            # Datos de la matriz de datos del edificio para valores de comparación
                            pmv_summer_matrix = comparison_data['pmv_summer'].values[0]
                            ppd_summer_matrix = comparison_data['ppd_summer'].values[0]
                            pmv_winter_matrix = comparison_data['pmv_winter'].values[0]
                            ppd_winter_matrix = comparison_data['ppd_winter'].values[0]

                            # Diferencias entre los valores de entrada y los de la matriz
                            difference_pmv_summer = pmv_summer_bucket - pmv_summer_matrix
                            difference_ppd_summer = ppd_summer_bucket - ppd_summer_matrix
                            difference_pmv_winter = pmv_winter_bucket - pmv_winter_matrix
                            difference_ppd_winter = ppd_winter_bucket - ppd_winter_matrix

                            # Agrega los resultados a las listas
                            hours_list.append(hour)
                            difference_pmv_summer_list.append(difference_pmv_summer)
                            difference_ppd_summer_list.append(difference_ppd_summer)
                            difference_pmv_winter_list.append(difference_pmv_winter)
                            difference_ppd_winter_list.append(difference_ppd_winter)

                    # Crea un DataFrame con los resultados        
                    results_df = pd.DataFrame({
                    'hour': hours_list,
                    'difference_pmv_summer': difference_pmv_summer_list,
                    'difference_ppd_summer': difference_ppd_summer_list,
                    'difference_pmv_winter': difference_pmv_winter_list,
                    'difference_ppd_winter': difference_ppd_winter_list})
                    results_df = bucket_data.merge(results_df, on='hour', how='left')
                results = pd.concat([results, results_df], ignore_index=True)
        return results if not results.empty else data_prepare_bucket

    
    def _set_in_start(self, X, y=None):
        """Set initial values for the model.
        
        Args:
            X (pd.DataFrame): Input data.
            y (pd.DataFrame, optional): Target data. Defaults to None.
        """
        logger.info("Setting initial values for PMVPPDAnalysis model")
        # Implement logic to set initial values here
        return self
    
    def _calculate_plant_status(self, df: pd.DataFrame, percentage_threshold: float=0.4) -> pd.DataFrame:
            """Calculates when a plant is considered 'on' based on the percentage of fancoils that are on.
            
            Args:
            df (pd.DataFrame): DataFrame containing 'floor' and 'general_condition' columns.
            percentage_threshold (float): Percentage of fancoils that need to be on for the plant to be considered on.
            
            Returns:
            pd.DataFrame: DataFrame (_"time") with plant status "plant_on".
            """
            plant_status = df.copy()
            plant_status = plant_status[plant_status['_field'] == "general_condition"]
            plant_status = plant_status.groupby(['floor', "_time"])['_value'].mean().reset_index()
            plant_status.loc[:,'plant_on'] = plant_status['_value'] > percentage_threshold
            plant_status_true = plant_status[plant_status['plant_on']][["_time", "floor"]]
            return plant_status_true
    
    
    def _calculate_data_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculates the system matrix based on the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing system data.
        
        Returns:
            pd.DataFrame: DataFrame with the system matrix.
        """
        data_matrix_list = []
        bucket_mean_list = []
        for bucket in X.bucket.unique():
            X_bucket = X[X.bucket == bucket]
            if X_bucket.empty:
                logger.warning(f"No data found for bucket {bucket}. Skipping fitting for this bucket.")
                continue
            else:
                plant_status = self._calculate_plant_status(X_bucket)
                if plant_status.empty:
                    logger.warning(f"The building {bucket} does not appear to have active plants for the established date range, cannot evaluate.")
                    continue
                plant_on_times = plant_status['_time']
                X_filtered = X_bucket.copy()
                X_filtered = X_bucket[X_bucket['_time'].isin(plant_on_times)][["_field", "floor", "_time", "_value"]]
                # Umbrales
                thresholds_co2 = self.thresholds.get("room_co2", (100, 1000))
                thresholds_humidity = self.thresholds.get("room_humidity", (5, 45))
                thresholds_temperature = self.thresholds.get("room_temperature", (5, 90))
                # Valores sin filtrar
                humidity = X_filtered[X_filtered['_field'] == "room_humidity"]
                co2 = X_filtered[X_filtered['_field'] == "room_co2"]
                temperature = X_filtered[X_filtered['_field'] == "room_temperature"]
                # Valores filtrados
                co2 = co2[(co2['_value'] >= thresholds_co2[0]) & (co2['_value'] <= thresholds_co2[1])]
                temperature = temperature[(temperature['_value'] >= thresholds_temperature[0]) & (temperature['_value'] <= thresholds_temperature[1])]
                humidity = humidity[(humidity['_value'] >= thresholds_humidity[0]) & (humidity['_value'] <= thresholds_humidity[1])]
                # Medias
                mean_humidity = humidity[humidity['_field'] == "room_humidity"].groupby(['floor', "_time"])['_value'].mean().reset_index()
                mean_temperature = temperature[temperature['_field'] == "room_temperature"].groupby(['floor', "_time"])['_value'].mean().reset_index()
                mean_co2 = co2[co2['_field'] == "room_co2"].groupby(['floor', "_time"])['_value'].mean().reset_index()
                # Extraer la hora del día
                mean_humidity.loc[:, 'hour'] = mean_humidity['_time'].dt.hour
                mean_temperature.loc[:, 'hour'] = mean_temperature['_time'].dt.hour
                mean_co2.loc[:, 'hour'] = mean_co2['_time'].dt.hour

                # Calcular la media por planta y hora
                mean_humidity = mean_humidity.groupby(['floor', 'hour'])['_value'].mean().reset_index()
                mean_temperature = mean_temperature.groupby(['floor', 'hour'])['_value'].mean().reset_index()
                mean_co2 = mean_co2.groupby(['floor', 'hour'])['_value'].mean().reset_index()
                
                # Unir los DataFrames
                mean_data = pd.merge(mean_humidity, mean_temperature, on=['floor', 'hour'], suffixes=('_humidity', '_temperature'))
                mean_data = pd.merge(mean_data, mean_co2, on=['floor', 'hour'])
                mean_data = mean_data.rename(columns={'_value': '_value_co2'})
                
                # Renombrar las columnas antes de la unión
                mean_data = mean_data.rename(columns={
                    '_value_humidity': 'humidity',
                    '_value_temperature': 'temperature',
                    '_value_co2': 'co2'
                })
                mean_data_ = mean_data.copy()
                mean_data_.loc[:, "bucket"] = bucket 
                
                # Aplicar la función pmv_ppd_extimate a cada fila
                mean_data_[['pmv_summer', 'ppd_summer', 'tsens_summer', 'pmv_winter', 'ppd_winter', 'tsens_winter']] = mean_data.apply(lambda row: self._pmv_ppd_extimate(temp=row['temperature'], rh=row['humidity']), axis=1, result_type='expand')
                data_matrix_list.append(mean_data_)                 
                              
                # Calcular la media por hora para cada planta
                mean_data_bucket = mean_data_.groupby(['hour']).agg({col: 'mean' for col in mean_data_.columns if mean_data_[col].dtype in ['int64', 'float64']}).reset_index()
                
                mean_data_bucket[['pmv_summer', 'ppd_summer', 'tsens_summer', 'pmv_winter', 'ppd_winter', 'tsens_winter']] = mean_data_bucket.apply(lambda row: self._pmv_ppd_extimate(temp=row['temperature'], rh=row['humidity']), axis=1, result_type='expand')
                
                mean_data_bucket.loc[:, "bucket"] = bucket
                bucket_mean_list.append(mean_data_bucket)
                
        logger.info("Calculating system matrix dictionary")
        
        return pd.concat(data_matrix_list) if data_matrix_list else pd.DataFrame(), pd.concat(bucket_mean_list) if bucket_mean_list else pd.DataFrame()
    
    def _pmv_ppd_extimate(self, temp: float, rh: float) -> float:
        """Estimates PMV and PPD values based on temperature and relative humidity.

        Args:
            temp (float): Temperature in Celsius.
            rh (float): Relative humidity in %.

        Returns:
            float: _description_
        """
        pmv_ppd_summer = pmv_ppd_iso(
            tdb=temp,  
            tr=temp,  # Mean radiant temperature in Celsius
            vr=0,  # Air velocity in m/s
            rh=rh,  # Relative humidity in %
            met=1.1,  # Metabolic rate in met
            clo=0.5,  # Clothing insulation in clo
            model='7730-2005'  # Standard to use for PMV/PPD calculation
        )
        pmv_ppd_winter = pmv_ppd_iso(
            tdb=temp,  
            tr=temp,  # Mean radiant temperature in Celsius
            vr=0,  # Air velocity in m/s
            rh=rh,  # Relative humidity in %
            met=1.1,  # Metabolic rate in met
            clo=1.1,  # Clothing insulation in clo
            model='7730-2005'  # Standard to use for PMV/PPD calculation
        )
        return pmv_ppd_summer.pmv, pmv_ppd_summer.ppd, pmv_ppd_summer.tsv, pmv_ppd_winter.pmv, pmv_ppd_winter.ppd, pmv_ppd_winter.tsv
        
