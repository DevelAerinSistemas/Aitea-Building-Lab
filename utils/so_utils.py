'''
 # @ Project: AItea-Brain-Lite
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-05-23 09:53:30
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-05-27 16:09:03
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 '''

'''
 # @ Project: AIteaBuilding-Lab
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-05-23 09:53:30
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-05-23 09:53:32
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 '''

import base64
import subprocess
from loguru import logger



def create_so(model_path: str) -> None:
    with open(model_path, "rb") as f:
        pipe_data = f.read()
        pipe_data_base64 = base64.b64encode(pipe_data).decode("utf-8")
    executor_code = f"""
import base64
import dill
import pandas as pd  # Import pandas
import logging
def load_pipeline():
    pipe_data_base64 = "{pipe_data_base64}"
    pipe_data = base64.b64decode(pipe_data_base64)
    pipeline = dill.loads(pipe_data)
    return pipeline
pipeline = load_pipeline()
class PipeExecutor:
    '''
        PipeExecutor: A class for executing pre-trained pipelines.

        This class loads a pre-trained pipeline from a serialized file ('.pkl')
        and provides methods for executing the pipeline, such as prediction and
        result extraction.

        Methods:
            get_query(): Returns the training query associated with the pipeline.
            fit_predict(X, y=None): Fits the pipeline to the input data and returns predictions.
            predict(X): Returns predictions based on the input data.
            get_results(Prediction): Extracts and returns results from the pipeline's steps.
            get_matrix(): Extracts and returns data matrices from the pipeline's steps.
            get_pipe_info(): Returns information about the pipeline's structure and components.
    '''
    def __init__(self):
        self.pipe = pipeline
    
    def get_query(self):
        query = self.pipe.get("training_query")
        if query:
            return query.copy()  # Returns a copy to avoid modifying the original
        return None

    def fit_predict(self, X: pd.DataFrame, y: pd.DataFrame = None):
        pipe_core = self.pipe.get("pipe")
        return pipe_core.fit_predict(X, y)
        
    def predict(self, X: pd.DataFrame):
        pipe_core = self.pipe.get("pipe")
        return pipe_core.predict(X)
    
    def predict_and_refit(self, X: pd.DataFrame):
        pipe_core = self.pipe.get("pipe")
        return pipe_core.predict_and_refit(X)
        
    def get_results(self) -> pd.DataFrame:
        result = list()
        pipe_core = self.pipe.get("pipe")
        try:
            for pipe_section in pipe_core.steps:
                try:
                    one_result = pipe_section[1].get_results()
                except Exception as err:
                    logging.warning('Possibly the pipe section does not have a get results')
                else:
                    result.append(one_result)
        except Exception as err:
            logging.error('Error in get_results')
        return result
    
    def update_parameters(self, **attributes):
        pipe_core = self.pipe.get("pipe")
        for pipe_section in pipe_core.steps:
            try:
                pipe_section[1].update_parameters(attributes)
            except Exception as err:
                logging.warning(f'Error updating parameters')
        
    def get_matrix(self) -> list:
        matrix = list()
        pipe_core = self.pipe.get("pipe")
        try:
            for pipe_section in pipe_core.steps:
                try:
                    one_matrix = pipe_section[1].get_matrix()
                except Exception as err:
                    logging.warning("Possibly the pipe section does not have a get results")
                else:
                    matrix.append(one_matrix)
        except Exception as err:
            logging.error(f"Error in get_results")
        return matrix
    
    def get_info(self) -> dict:
        info = dict()
        pipe_core = self.pipe.get("pipe")
        for pipe_section in pipe_core.steps:
            try:
                one_info = pipe_section[1].get_info()
            except Exception as err:
                logging.warning("Possibly the pipe section does not have a get info")
            else:
                info[pipe_section[0]] = one_info
        return info
    
    def get_all_attributes(self) -> dict:
        info = dict()
        pipe_core = self.pipe.get("pipe")
        for pipe_section in pipe_core.steps:
            try:
                one_info = pipe_section[1].get_all_attributes()
            except Exception as err:
                logging.warning("Possibly the pipe section does not have a get info")
            else:
                info[pipe_section[0]] = one_info
        return info
    """
    
    name_so = model_path.split("/")[1].split(".")[0]
    name_py = name_so + ".py"
    
    with open(name_py, "w") as f:
            f.write(executor_code)

    command = [
        "/opt/VirtualEnv/virtualAiteaBuildingLab/bin/nuitka",
        "--module", name_py,
        "--include-package=models_warehouse",
        "--include-package=metaclass",
        "--include-package=utils",
        "--show-modules",
        "--output-dir=lib",
        "--remove-output"
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f" Output: {result.stdout}")
        if result.stderr:
            logger.error(f"Error Output: {result.stderr}")
    except subprocess.CalledProcessError as err:
        logger.error(f"Error Output: {err}")
    finally:
        # Clean up the temporary file
        import os
        import glob
        if os.path.exists(name_py):
            os.remove(name_py)
            files = glob.glob("lib/*.pyi")
            for file in files:
                os.remove(file)
            logger.info("Temporary file removed.")


if __name__ == "__main__":
    # Example usage
    create_so("training_models/consumption_analysis.pkl")
    logger.info("SO file created successfully.")
