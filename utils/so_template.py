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

    def get_all_class_attributes(self):
        pipe_core = self.pipe.get("pipe")
        attributes = dict()
        for pipe_section in pipe_core.steps:
            pipe_name = pipe_section[0]
            one_pipe = pipe_section[1]
            attributes[pipe_name] = one_pipe.__dict__
        return attributes