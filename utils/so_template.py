from dotenv import load_dotenv
load_dotenv()
import os
import base64
import dill
import pandas as pd  # Import pandas
import logging

from utils.file_utils import load_json_file

try:
    from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
    from aitea_connectors.connectors.postgresql_connector import PostgreSQLConnector
    AITEA_CONNECTORS = True
except ImportError:
    logging.warning(f"Aitea Connectors are not available. Uncomplete functionality: only local files as a valid data source.")
    AITEA_CONNECTORS = False

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
    def __init__(self, connections_path: str):
        self.pipe = pipeline
        if os.path.exists(connections_path):
            os.environ["CONNECTIONS_PATH"] = connections_path
        self.connections = self._create_connections()
    
    def _create_connections(self):
        connections = {}
        if AITEA_CONNECTORS:
            for conn_name in set(self.pipe.get("training_info"))-{"local"}:
                if conn_name == "influxdb":
                    connections[conn_name] = {"connector": InfluxDBConnector()}
                elif conn_name == "postgresql":
                    connections[conn_name] = {"connector": PostgreSQLConnector()}
                connections[conn_name].update(zip(("connection_status","connection_client"),connections[conn_name]["connector"].connect()))
        return connections

    def get_connections(self):
        return self.connections

    def get_influxdb_connector(self):
        return self.connections.get("influxdb")
    
    def get_postgresql_connector(self):
        return self.connections.get("postgresql")

    def get_influxdb_query(self):
        query = self.pipe.get("training_info",{}).get("influxdb")
        if query:
            return query.copy()  # Returns a copy to avoid modifying the original
        return None
    
    def get_postgresql_query(self):
        query = self.pipe.get("training_info",{}).get("postgresql")
        if query:
            return query.copy()  # Returns a copy to avoid modifying the original
        return None
    
    def get_datafiles(self):
        query = self.pipe.get("training_info",{}).get("local")
        if query:
            return query.copy()  # Returns a copy to avoid modifying the original
        return None

    def get_training_info(self):
        query = self.pipe.get("training_info")
        if query:
            return query.copy()  # Returns a copy to avoid modifying the original
        return None

    def fit_predict(self, X: pd.DataFrame, y: pd.DataFrame = None):
        pipe_core = self.pipe.get("pipe")
        return pipe_core.fit_predict(X, y)
        
    def predict(self, X: pd.DataFrame):
        pipe_core = self.pipe.get("pipe")
        return pipe_core.predict(X)
    
    def predict_and_partial_fit(self, X: pd.DataFrame):
        returned = None
        pipe_core = self.pipe.get("pipe")
        try:
            returned = pipe_core.predict_and_partial_fit(X)
        except Exception as err:
            logging.error(f"Error in predict_and_partial_fit: It is possible that the analytics cannot be partially fitted or the function is not defined. Original error: {err}")
            returned = self.predict(X)
        else:
            logging.info("Predict and partial fit executed successfully")
        return returned 
        
    def get_results(self) -> pd.DataFrame:
        result = list()
        pipe_core = self.pipe.get("pipe")
        try:
            for pipe_section in pipe_core.steps:
                try:
                    one_result = pipe_section[1].get_results()
                except Exception as err:
                    logging.warning("Possibly the pipe section does not have a 'get_results' method")
                else:
                    result.append(one_result)
        except Exception as err:
            logging.error('Error in get_results')
        return result
    
    def update_parameters(self, **attributes):
        pipe_core = self.pipe.get("pipe")
        for pipe_section in pipe_core.steps:
            try:
                pipe_section[1].update_parameters(**attributes)
            except Exception as err:
                logging.warning(f'Error updating parameters')
            else:
                logging.info(f'Parameters updated for {{attributes}}')
        
    def get_matrix(self) -> list:
        matrix = list()
        pipe_core = self.pipe.get("pipe")
        try:
            for pipe_section in pipe_core.steps:
                try:
                    one_matrix = pipe_section[1].get_matrix()
                except Exception as err:
                    logging.warning("Possibly the pipe section does not have a 'get_matrix' method")
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
                logging.warning("Possibly the pipe section does not have a 'get_info' method")
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
                logging.warning("Possibly the pipe section does not have a 'get_all_attributes' method")
            else:
                info[pipe_section[0]] = one_info
        return info

    def get_all_class_attributes(self):
        pipe_core = self.pipe.get("pipe")
        attributes = dict()
        for pipe_section in pipe_core.steps:
            pipe_name = pipe_section[0]
            one_pipe = pipe_section[1]
            attributes[pipe_name] = one_pipe.__dict__
        return attributes