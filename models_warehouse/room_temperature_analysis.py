'''
 # @ Author: Jose Luis Blanco Garcia-Moreno <joseluis.blanco@aitea.tech>
 # @ Create Time: 2024-09-09 12:30:38
 # @ Modified by: Jose LUis Blanco Garcia-Moreno
 # @ Modified time: 2024-09-09 12:31:15
 # @ Proyect: Aitea Building Lab
 # @ Description: Room temperature analysis class
 # @ Copyright (c) 2024: Departamento de I+D. Aitea Tech
 '''

from typing import Any
import numpy as np

from metaclass.templates import MetaTransform

class RoomTemperatureTransform(MetaTransform):
    def __init__(self, datetime_name: str = "_time", room_temperature_name: str = "room_temperature", star_stop_condition: str ="general_condition"):
        """This transformation separates the dataframe into rising, plateau, and falling room temperature.

        Args:
            datetime_name (str, optional): Datetime name. Defaults to "_time".
            room_temperature_name (str, optional): room temperature attribute name. Defaults to "room_temperature".
            star_stop_condition (str, optional): Star or stop attribute name. Defaults to "general_condition    ".
        """
    
    def fit(self, X: Any, y: Any = None) -> None:
        pass
    
    def transform(self, X: Any) -> np.ndarray:
        pass
        
    def fit_transform(self, X: Any, y: None) -> np.ndarray:
        pass


if __name__ == "__main__":
    rt =  RoomTemperatureTransform()