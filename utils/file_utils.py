'''
 # @ Project: AItea-Brain-Lite
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-03-20
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-05-15
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
'''



import json
from json.decoder import JSONDecodeError
from  loguru import logger

@logger.catch
def load_json_file(path: str) -> dict:
    """Load json file

    Args:
        path (str): File path

    Returns:
        dict: Json as a dictionary 
    """
    json_file = None
    try:
        f = open(path, mode="r", encoding="utf-8")
        json_file = json.loads(f.read())
        f.close()
        logger.debug(f"Read file: {path}")
    except Exception as e:
        logger.error(
            f"Error reading file {path}: {e}"
        )
    return json_file

