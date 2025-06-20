'''
 # @ Author: Héctor Bermúdez Castro <hector.bermudez@aitea.tech>
 # @ Create Time: 2025-06-20
 # @ Modified by: Héctor Bermúdez Castro
 # @ Modified time: 2025-06-20
 # @ Project: Aitea Building Lab
 # @ Description: Temperature Reach Analysis
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from loguru import logger

import numpy as np
from scipy.signal import argrelextrema
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import warnings

from metaclass.templates import MetaTransform

LIBRARY_VERSION = datetime.now().strftime("%Y_%m_%d_%H_%M")

class TemperatureReachAnalytic(MetaTransform):

    def __init__(self, **parameters):
        """Initializes class"""
        super().__init__()
        self.parameters = parameters
        self.results_dictionary = {}

        self.bucket_list = self.get_available_buckets()
        self.event_name = self.parameters.get("event_name")
        self.delta_hours = self.parameters.get("windows_size_hours")
        self.start_time_hour = self.parameters.get("start_time_hour")
        self.start_time_minute = self.parameters.get("start_time_minute")
        self.utc = ZoneInfo(self.parameters.get("zone_info"))
        self.now_time = datetime.now(self.utc)
    
    # ABSTRACT METHODS

    def fit(self, X: Any, y: Any = None) -> None:
        pass
    
    def transform(self, X: Any) -> np.ndarray:
        pass
    
    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        pass

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
    def result_processor(self,  pre_result: pd.DataFrame) -> pd.DataFrame:
        """Method used to pre-process results. It returns a dataframe with the results.

        Args:
            results (pd.DataFrame): query results from the InfluxDB in a pandas DataFrame

        Returns:
            pd.DataFrame: dataframe with the results 
        """

        columns = ["time","value","field","building","floor","module","zone","client"]
        results = pre_result.rename(columns={
            "_time": "time",
            "_value":"value",
            "_field":"field"
        })[columns]

        return results

    @logger.catch
    def result_analyzer(self, pre_result: pd.DataFrame) -> pd.DataFrame:
        """Analyze the results

        Args:
            pre_result (pd.DataFrame): Dataframe from which to compute analytics 
        
        Returns:
            pd.DataFrame: dataframe with the results 
        """

        f_values = pre_result["field"].unique()
        dict_results = {
            f:pre_result[pre_result["field"]==f].drop(columns=["field"]).reset_index(drop=True).rename(columns={"value":f}) 
            for f in f_values
        }
        group_columns = ["floor","module","zone","client"]
        for f in dict_results:
            for i in range(len(group_columns),-1,-1):
                dict_results[f] = dict_results[f].groupby(["building"]+group_columns[:i]+["time"])[f].mean().reset_index()
        results = pd.merge(
            dict_results[f_values[0]],
            dict_results[f_values[1]],
            on=["building","time"]
        )
        results["time"] = results["time"].apply(lambda x: pd.Timestamp(x).tz_convert("Europe/Madrid"))
        
        building_analysis = results.groupby("building").apply(
            lambda subdf: results["time"].iloc[self.get_local_extrema(subdf,"room_temperature")].values,
            include_groups=False
        ).rename("times").reset_index()
        building_analysis["times"] = building_analysis["times"].apply(lambda l: [pd.Timestamp(li).tz_localize("UTC").tz_convert("Europe/Madrid") for li in l])

        # results.to_csv("./analytics/temperature_reach_analytic.csv",index=False)
        # building_analysis.to_csv("./analytics/temperature_reach_analytic.csv",index=False)

        try:
            now_time = self.now_time
            start_temp = datetime(year=now_time.year,month=now_time.month,day=now_time.day,hour=self.start_time_hour,minute=self.start_time_minute,tzinfo=now_time.tzinfo)
            end_temp = start_temp + timedelta(hours=2)
            building_analysis["boot"] = building_analysis["times"].apply(lambda l: [li if (li>pd.Timestamp(start_temp) and li<pd.Timestamp(end_temp)) else pd.Timestamp(start_temp) for li in l][0])
            building_analysis = pd.merge(
                building_analysis,
                results.groupby("building").apply(
                    lambda subdf: abs(subdf["room_temperature"].iloc[-1]-subdf[subdf["time"]==building_analysis[building_analysis["building"]==subdf.name]["boot"].loc[0]]["room_temperature"].iloc[0]) > abs(subdf["room_temperature"].iloc[-1]-subdf["setpoint_temperature"].iloc[-1]),
                    include_groups=False
                ).rename("ok_temp"), 
                on="building"
            )
        except Exception as err:
            if "indexer is out-of-bounds" in f"{err}":
                err = "Results from booting-time automated computation were not inside expected time window"
            logger.warning(f"""Error: {err}""")
        finally:
            building_analysis = pd.merge(
                building_analysis,
                results.groupby("building").last(),
                on=["building"]
            ).drop(columns=["times","boot"])
            building_analysis["ok_temp"] = "boot" in building_analysis
            
        return building_analysis

    def get_local_extrema(self,data: pd.DataFrame, column: str, perc:float=0.3) -> list:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            column (str): _description_
            perc (float, optional): _description_. Defaults to 0.3.

        Returns:
            _type_: _description_
        """
        
        order = int(len(data)*perc) # number of points for computing local extrema
        mm = np.sort(np.concatenate((
            argrelextrema(data[column].values,np.greater,order=order)[0],
            argrelextrema(data[column].values,np.less,order=order)[0]
        ))) # local extrema indices

        return list(mm)

    @logger.catch
    def main_engine(self) -> None:  
        """Method that manages the main engine for the analytic
        """

        logger.info(f"""Initiating analysis""")
        results = pd.DataFrame()

        for bucket in self.bucket_list:
            bucket_results = self.influx_results(bucket=bucket)
            if bucket_results is None:
                logger.error(f"""Something went wrong when searching data in bucket '{bucket}'""")
            elif not bucket_results.empty:
                bucket_results["building"] = bucket
                results = pd.concat([results,bucket_results])
            else:
                logger.warning(f"""No results available for bucket '{bucket}'""")
        
        results = self.result_processor(results)
        results = self.result_analyzer(results)

        logger.info(f"""Results:\n{results}""")

        if results is None:
            logger.error(f"""No results available, no analytic messages sent""")
        else:
            if not results.empty:
                for building, building_report in results.set_index("building").to_dict(orient="index").items():

                    system_info = {}
                    system_info["analytics"] = self.event_name
                    system_info["building"] = building
                    system_info["first_date"] = self.now_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                    system_info["date"] = self.now_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                    status = "no_success"
                    if building_report["ok_temp"]:
                        status = "success"
                    args = {
                        "building": building,
                        "setpoint_temperature": f"""{building_report["setpoint_temperature"]:.2f}""",
                        "room_temperature": f"""{building_report["room_temperature"]:.2f}""",
                        "success_status": status
                    }
                    system_info["message"] = self.event_manager.get_render("building_optimum_temperature_report", args)
                    system_info["priority"] = 1
                    system_info["event_type"] = "event"
                    
                    self.launch_event(to_kafka=True, **system_info)
            else:
                logger.error(f"""No results available, no analytic messages sent""")
            
        logger.info(f"""Analysis has finished""")