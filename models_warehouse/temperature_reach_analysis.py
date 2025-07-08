'''
 # @ Author: Héctor Bermúdez Castro <hector.bermudez@aitea.tech>
 # @ Create Time: 2025-06-20
 # @ Modified by: Héctor Bermúdez Castro
 # @ Modified time: 2025-06-20
 # @ Project: Aitea Building Lab
 # @ Description: Temperature Reach Analysis
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from utils.logger_config import get_logger
logger = get_logger()
import inspect

import numpy as np
from scipy.signal import argrelextrema
import pandas as pd

from datetime import datetime, timedelta
import warnings

from metaclass.templates import MetaTransform

from zoneinfo import ZoneInfo
from typing import Any, Dict, List

LIBRARY_VERSION = datetime.now().strftime("%Y_%m_%d_%H_%M")

class TemperatureReachTransform(MetaTransform):

    @logger.catch
    def __init__(self, **parameters):
        """Initializes class"""
        super().__init__()
        self.parameters = parameters
        self.results_dictionary = {}

        self.event_name = self.parameters.get("event_name")
        self.delta_hours = self.parameters.get("windows_size_hours")
        self.start_time_hour = self.parameters.get("start_time_hour")
        self.start_time_minute = self.parameters.get("start_time_minute")
        self.utc = self.parameters.get("zone_info")
        self.now_time = datetime.now(ZoneInfo(self.utc))
    
    # ABSTRACT METHODS

    @logger.catch
    def fit(self, X: Any, y: Any = None) -> None:
        """
        This transform does not learn from data, so fit does nothing.
        """
        # logger.info(f"{self.__class__.__name__} fit method called, no operation performed.")
        pass
    
    @logger.catch
    def transform(self, X: Any) -> np.ndarray:
        """
        Main transformation logic. Assumes X is the combined DataFrame from all buckets.
        """
        logger.info(f"Initiating analysis in {self.__class__.__name__}")
        if X is None or X.empty:
            logger.warning("Input DataFrame X is empty or None. No analysis will be performed.")
            self.results_dictionary["building_analysis"] = pd.DataFrame()
            return pd.DataFrame()
        
        # 1. Process results
        processed_df = self._result_processor(X)

        if processed_df.empty:
            logger.warning("Result processor returned an empty DataFrame. No further analysis.")
            self.results_dictionary["building_analysis"] = pd.DataFrame()
            return pd.DataFrame()
            
        # 2. Analyze results
        analyzed_results_df = self._result_analyzer(processed_df)
        
        self.results_dictionary["building_analysis"] = analyzed_results_df
        logger.info(f"Analysis Results:\n{analyzed_results_df}")

        # 3. Send events (adapted from main_engine)
        if analyzed_results_df is None or analyzed_results_df.empty:
            logger.error("No analysis results available, no analytic messages sent.")
        else:
            for building, building_report in analyzed_results_df.set_index("building").to_dict(orient="index").items():
                system_info: Dict[str, Any] = {}
                system_info["analytics"] = self.event_name
                system_info["building"] = building
                # Using self.now_time which is initialized with the correct timezone
                current_time_str = self.now_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                system_info["first_date"] = current_time_str
                system_info["date"] = current_time_str
                
                status = "no_success"
                # Ensure 'ok_temp' key exists and is True
                if building_report.get("ok_temp", False): 
                    status = "success"
                
                args = {
                    "building": building,
                    "setpoint_temperature": f"""{building_report.get("setpoint_temperature", float('nan')):.2f}""",
                    "room_temperature": f"""{building_report.get("room_temperature", float('nan')):.2f}""",
                    "success_status": status
                }
                # These methods (event_manager.get_render, launch_event) are assumed to be
                # available on self, possibly via MetaTransform or injection.
                try:
                    system_info["message"] = self.event_manager.get_render("building_optimum_temperature_report", args)
                    system_info["priority"] = 1
                    system_info["event_type"] = "event"
                    self.launch_event(to_kafka=True, **system_info)
                except AttributeError as e:
                    logger.error(f"Failed to send event for {building}. Missing event infrastructure: {e}")
                except Exception as e:
                    logger.error(f"An error occurred while preparing or sending event for {building}: {e}")
        
        logger.info(f"{self.__class__.__name__} analysis has finished.")
        return analyzed_results_df

    @logger.catch    
    def fit_transform(self, X: Any, y: Any = None) -> pd.DataFrame:
        """
        Applies fit and then transform.
        """
        self.fit(X,y)
        return self.transform(X)

    def get_info(self) -> str:
        return f"{self.__class__.__name__} v.{LIBRARY_VERSION}"
    
    def get_all_attributes(self) -> dict:
        return self.parameters

    def get_results(self) -> dict:
        return self.results_dictionary
    
    # NON ABSTRACT METHODS

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
    def generate_query_params(self):
        """TBD"""
        time_start = self.now_time - timedelta(hours=self.delta_hours)
        time_start = time_start.isoformat()
        time_now = self.now_time.isoformat()
        
        return {"time_start": time_start, "time_now": time_now}

    @logger.catch
    def _result_processor(self,  pre_result: pd.DataFrame) -> pd.DataFrame:
        """Method used to pre-process results. It returns a dataframe with the results.

        Args:
            results (pd.DataFrame): query results from the InfluxDB in a pandas DataFrame

        Returns:
            pd.DataFrame: dataframe with the results 
        """

        columns = ["time","value","field","building","floor","module","zone","client"]
        
        if pre_result.empty:
            logger.warning(f"Input to function '{_inspect.currentframe().f_code.co_name}' is empty. Returning empty DataFrame.")
            return pd.DataFrame(columns=columns)
        
        results = pre_result.rename(
            columns={
                "_time": "time",
                "_value":"value",
                "_field":"field"
            }
        )

        final_columns = [col for col in columns if col in results.columns]
        missing_cols = set(columns) - set(final_columns)
        if missing_cols:
            logger.warning(f"Columns missing from input for function '{inspect.currentframe().f_code.co_name}': {missing_cols}. They will not be in the output.")

        return results[final_columns]
    
    @logger.catch
    def _get_local_extrema(self,data: pd.DataFrame, column: str, perc:float=0.3) -> list:
        """_summary_

        Args:
            data (pd.DataFrame): DataFrame containing the data.
            column (str): Column name to find extrema in.
            perc (float, optional): Percentage of data points to define order for extrema. Defaults to 0.3.

        Returns:
            list: List of indices of local extrema.
        """
        if data.empty or column not in data.columns or len(data) == 0:
            logger.warning(f"Cannot get local extrema. Data is empty, column '{column}' missing, or data length is zero.")
            return []
        order = int(len(data)*perc) # number of points for computing local extrema
        if order == 0 and len(data) > 0: # Order is at least 1 if data exists, to avoid argrelextrema error
            order = 1 
        if order <= 0 :
            logger.warning(f"Order for argrelextrema is {order}. Cannot compute_extrema. Returning empty list.")
            return []
        try:
            maxima = argrelextrema(data[column].values, np.greater, order=order)[0]
            minima = argrelextrema(data[column].values, np.less, order=order)[0]
            mm = np.sort(np.concatenate((maxima, minima))) 
            return list(mm)
        except Exceptions as e:
            logger.error(f"Error computing local extrema for column '{column}': {e}")
            return []

    @logger.catch
    def _result_analyzer(self, pre_result: pd.DataFrame) -> pd.DataFrame:
        """Analyze the results

        Args:
            pre_result (pd.DataFrame): Dataframe from which to compute analytics. Expected to be output of _result_processor. 
        
        Returns:
            pd.DataFrame: dataframe with the analyzed results (building_analysis)
        """

        if pre_result.empty:
            logger.warning(f"Input to function '{inspect.currentframe().f_code.co_name}' is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        if "field" not in pre_result.columns:
            logger.error(f"'field' column is missing in pre_result for function '{inspect.currentframe().f_code.co_name}'. Returning empty DataFrame.")
            return pd.DataFrame()

        f_values = pre_result["field"].unique()
        if len(f_values) < 2:
            logger.error(f"Requires at least two unique 'field' values for analysis, found {len(f_values)}: {f_values}. Cannot proceed with merge step. Returning empty DataFrame")
            return pd.DataFrame() 

        dict_results = {}
        for f in f_values:
            cols_to_drop = ["field"]
            df_field = pre_result[pre_result["field"]==f].drop(columns=cols_to_drop, errors='ignore').reset_index(drop=True)
            dict_results[f] = df_field.rename(columns={"value":f}) 
            
        
        group_columns = ["floor","module","zone","client"]
        if dict_results: 
            available_cols_for_grouping = list(dict_results[f_values[0]].columns)
            valid_group_columns = [col for col in group_columns if col in available_cols_for_grouping]
        else: 
            valid_group_columns = []
        for f in dict_results:
            if "building" not in dict_results[f].columns:
                logger.error(f"Column 'building' missing in data for field '{f}'. Skipping grouping for this field.")
                continue
            for i in range(len(valid_group_columns),-1,-1):
                current_grouping_cols = ["building"] + valid_group_columns[:i] + ["time"]
                # Check if f (the value column) exists before grouping
                if f not in dict_results[f].columns:
                    logger.error(f"Value column '{f}' missing in dataframe for field '{f}' before grouping. Skipping this group.")
                    continue
                dict_results[f] = dict_results[f].groupby(current_grouping_cols, as_index=False)[f].mean()
        try:
            if f_values[0] not in dict_results or f_values[1] not in dict_results:
                logger.error(f"Not enough processed fields in dict_results to perform merge. Needed '{f_values[0]}' and '{f_values[1]}'.")
                return pd.DataFrame()
        
            results = pd.merge(
                dict_results[f_values[0]],
                dict_results[f_values[1]],
                on=["building","time"]
            )
        except KeyError as e:
            logger.error(f"KeyError during merge, likely 'building' or 'time' missing after grouping: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error during merge: {e}")
            return pd.DataFrame()

        if results.empty:
            logger.warning("Merge operation resulted in an empty DataFrame.")
            return pd.DataFrame()

        try:
            results["time"] = pd.to_datetime(results["time"]).dt.tz_convert(self.utc)
        except Exception as e:
            logger.error(f"Error converting 'time' column to timezone '{self.utc}': {e}")
        
        if "room_temperature" not in results.columns or "building" not in results.columns:
            logger.error("'room_temperature' or 'building' column missing for building_analysis. Skipping this part.")
            building_analysis = pd.DataFrame(columns=["building"]) 
        else:
            building_analysis = results.groupby("building").apply(
                lambda subdf: results["time"].iloc[self.get_local_extrema(subdf,"room_temperature")].values,
                include_groups=False
            ).rename("times").reset_index()
            try:
                building_analysis["times"] = building_analysis["times"].apply(
                    lambda l: [
                        pd.Timestamp(li).tz_localize("UTC").tz_convert(self.utc) 
                        for li in l
                    ] if isinstance(l, (list, np.ndarray)) else []
                )
            except Exception as e:
                logger.error(f"Error during timezone conversion for 'times' in building_analysis: {e}")

        building_analysis_final = building_analysis.copy()
        building_analysis_final["ok_temp"] = False

        try:
            now_time = self.now_time
            start_temp_dt = datetime(
                year=now_time.year,
                month=now_time.month,
                day=now_time.day,
                hour=self.start_time_hour,
                minute=self.start_time_minute,
                tzinfo=now_time.tzinfo
            )
            end_temp_dt = start_temp_dt + timedelta(hours=2)

            if "times" not in building_analysis.columns or building_analysis["times"].empty:
                raise ValueError("'times' column is missing or empty in building_analysis. Cannot compute 'boot'.")

            start_temp_ts = pd.Timestamp(start_temp_dt)
            end_temp_ts = pd.Timestamp(end_temp_dt)

            building_analysis_final["boot"] = building_analysis["times"].apply(
                lambda l: [
                    li 
                    for li in l
                    if isinstance(li, pd.Timestamp) and start_temp_ts < li < end_temp_ts 
                ] or [start_temp_ts]
            ).apply(lambda x: x[0])

            if not all(col in results.columns for col in ["room_temperature", "setpoint_temperature"]):
                raise ValueError("Missing 'room_temperature' or 'setpoint_temperature' in results for 'ok_temp' calculation.")

            ok_temp_series = results.groupby("building").apply(
                lambda subdf: (
                    abs(subdf["room_temperature"].iloc[-1] - 
                        subdf[subdf["time"] == building_analysis_final.loc[building_analysis_final["building"] == subdf.name, "boot"].iloc[0]]["room_temperature"].iloc[0]) >
                    abs(subdf["room_temperature"].iloc[-1] - subdf["setpoint_temperature"].iloc[-1])
                    if not subdf[subdf["time"] == building_analysis_final.loc[building_analysis_final["building"] == subdf.name, "boot"].iloc[0]].empty # Check if boot time exists in subdf
                    else False # Default if boot time not found
                ),
                include_groups=False
            ).rename("ok_temp")

            building_analysis_final = pd.merge(
                building_analysis_final.drop(columns=["ok_temp"], errors="ignore"),
                ok_temp_series, 
                on="building",
                how="left"
            )
            building_analysis_final["ok_temp"] = building_analysis_final["ok_temp"].fillna(False)
        except Exception as err:
            if "indexer is out-of-bounds" in f"{err}" or isinstance(err,IndexError, KeyError):
                err = "Results from booting-time automated computation were not inside expected time window or data was missing."
            else:
                err_msg = f"Error during boot time analysis: {err}"
            logger.warning(f"""Error: {err}""")
        finally:
            if not results.empty and "building" in results.columns:
                last_values_per_building = results.groupby("building").last().reset_index()
                merge_cols = [
                    col 
                    for col in last_values_per_building.columns 
                    if col not in building_analysis_final.columns or col == "building"
                ]
                if not merge_cols or "building" not in merge_cols:
                    logger.warning(f"Few columns to merge from last_values_per_building in function {inspect.currentframe().f_code.co_name}'s finally block.")
                if "building" in last_values_per_building.columns:
                    building_analysis = pd.merge(
                        building_analysis_final,
                        last_values_per_building[merge_cols],
                        on=["building"],
                        how="left"
                    )
                else:
                    logger.warning(f"Results DataFrame is empty or 'building' column missing in function {inspect.currentframe().f_code.co_name}'s finally block; cannot merge last values.")
        
                building_analysis_final = building_analysis_final.drop(columns=["times", "boot"], errors='ignore')
                if "ok_temp" not in building_analysis_final.columns:
                    building_analysis_final["ok_temp"] = False

        return building_analysis_final