'''
 # @ Project: AItea-Building-Lab
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-05-29 10:35:08
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-05-29 10:35:12
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

from dotenv import load_dotenv
load_dotenv()
import os
import sys
# Geting the absolute path of the directory containing display.py (i.e., display/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Getting the absolute path of the project root (one level up from display/)
project_root = os.path.dirname(current_dir)
# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logger_config import get_logger
logger = get_logger()
# Getting global configuration
from utils.file_utils import load_json_file
global_config = load_json_file(os.getenv("CONFIG_PATH"))
LIB_DIR = os.path.join(project_root, global_config.get("libs_path"))
DATA_DIRS = [
    os.path.join(project_root, predicting_path)
    for predicting_path in global_config.get("predicting_paths")
]
from library_loader.so_library_loader import SOLibraryLoader  

try:
    from aitea_connectors.connectors.influxdb_connector import InfluxDBConnector
    from aitea_connectors.connectors.postgresql_connector import PostgreSQLConnector
    AITEA_CONNECTORS = True
except ImportError:
    logger.warning(f"⚠️ Aitea Connectors are not available. Uncomplete functionality: only local files as a valid data source.")
    AITEA_CONNECTORS = False

import time
import datetime
import streamlit as st
import plotly.graph_objects as go

@logger.catch()
def get_available_libraries() -> list:
    """Scans the library directory for .so files."""
    if not os.path.exists(LIB_DIR):
        st.error(f"Library directory not found: {LIB_DIR}")
        return [None]
    libs = [f.split(".")[0] for f in os.listdir(LIB_DIR) if f.endswith(".so")]
    return [None] + libs

@logger.catch()
def get_local_files() -> list:
    """Scans the data directories for supported files."""
    files = []
    for DATA_DIR in DATA_DIRS:
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                files.append(os.path.join(DATA_DIR, f))
        else:
            logger.warning(f"⚠️ Data directory '{DATA_DIR}' does not exist")
    return files

@logger.catch
def get_available_datasources() -> list:
    datasources = ["local"]
    if AITEA_CONNECTORS:
        datasources += ["influxdb", "postgresql"]
    return datasources 
        
@logger.catch()
def display_results(results: dict, result_matrix: dict, time_elapsed: int, library_name: int, table_or_graph: str) -> None:
    """Refactored function to display results from any source."""
    st.markdown(f"<span style='font-size: 24px;'>Time elapsed: {time_elapsed:.2f} seconds for {library_name}.</span>", unsafe_allow_html=True)
    
    if table_or_graph == "Table":
        st.markdown("---")
        create_table(result_matrix)
    else: # table_or_graph == "Graph"
        data = []
        for bucket, prediction in results.items():
            if hasattr(prediction, 'columns'):
                for column in prediction.columns:
                    data.append(go.Scatter(x=prediction.index, y=prediction[column], mode='lines', name=f"{bucket}-{column}"))
            else:
                logger.warning(f"Unexpected prediction format for bucket '{bucket}': {type(prediction)}")
        fig = go.Figure(data=data)
        fig.update_layout(title="Testing Results", xaxis_title="Time", yaxis_title="Value")
        st.plotly_chart(fig)

@logger.catch()
def create_table(data_dictionary: dict) -> None:
    st.markdown("---")
    for bucket, prediction in data_dictionary.items():
        for name, matrix in prediction[-1].items():
            st.markdown(f"### {bucket} - {name}")
            if hasattr(matrix, 'columns'):
                st.dataframe(matrix)
                st.markdown(f"### {bucket} - {name}. NaN table")
                numeric_matrix = matrix.select_dtypes(include=['number'])
                if not numeric_matrix.empty:
                    st.dataframe(numeric_matrix[numeric_matrix.isna().any(axis=1)])
                st.markdown(f"### {bucket} - {name}. dtypes")
                st.dataframe(matrix.dtypes.to_frame(name='Dtype'))
                st.markdown(f"### {bucket} - {name}. Description")
                st.dataframe(matrix.describe())
            else:
                logger.warning(f"Unexpected prediction format for bucket {bucket}: {type(matrix)}")

# Streamlit app
@logger.catch()
def main():
    st.title("SO Library Testing Interface")
    
    # Initialize session_state variables
    if "library_selected" not in st.session_state:
        st.session_state.library_selected = False
    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime.date(2025, 5, 1)
    if "stop_date" not in st.session_state:
        st.session_state.stop_date = datetime.date(2025, 5, 1)

    # Library Selection
    available_libraries = get_available_libraries()
    library_name = st.selectbox(
        "Select a library:",
        available_libraries,
        index=0,
        placeholder="Select a library..."
    )

    # Display library info (runs once per selection)
    if library_name and not st.session_state.get(f"info_shown_{library_name}", False):
        try:
            st.write(f"You selected: {library_name}")
            loader = SOLibraryLoader(library_name)
            info = loader.exec.get_info() if loader.exec else None
            if info:
                st.markdown("### Library Information:")
                for key, value in info.items():
                    st.markdown(f"**{key}:**")
                    st.markdown(f"<pre>{value}</pre>", unsafe_allow_html=True)
            else:
                st.warning("No information available for the selected library.")
            st.session_state[f"info_shown_{library_name}"] = True # Mark as shown for this specific library
        except Exception as e:
            logger.error(f"Error loading library info: {e}")
            st.error(f"Error loading library info: {e}")
    st.markdown("---")

    # Data Source Selection
    available_datasources = get_available_datasources()
    data_source = st.radio(
        "Select one Data Source:",
        available_datasources,
        horizontal=True
    )
    selected_file = None # Initialize variable
    if data_source == "local":
        # File selection
        local_files = get_local_files()
        if not local_files:
            st.warning(f"No data files found in the '{DATA_DIRS}' directories. Please add some files (e.g., .csv, .parquet).")
        selected_file = st.selectbox(
            "Select a local data file:",
            options=local_files,
            format_func=lambda x: os.path.basename(x) if x else "No files found", 
            placeholder="Select a file..."
        )
    else:
        # Date selection (for InfluxDB and PostgreSQL)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", st.session_state.start_date)
            start_time = st.time_input("Start Time", datetime.time(0, 0))
        with col2:
            stop_date = st.date_input("Stop Date", st.session_state.stop_date)
            stop_time = st.time_input("Stop Time", datetime.time(23, 59))
        st.session_state.start_date = start_date
        st.session_state.stop_date = stop_date        

    # Display format selection
    table_or_graph = st.radio(
        "Display Results As:", 
        ("Table", "Graph"), 
        index=1
    )

    # Execute Button
    if st.button("Execute Testing"):
        if not library_name:
            st.warning("Please select a library first.")
            return
        try:
            loader = SOLibraryLoader(library_name)
            time_init = time.time()
            results, result_matrix = None, None
            if data_source != "local":
                start_datetime = datetime.datetime.combine(start_date, start_time).isoformat(timespec='milliseconds') + 'Z'
                stop_datetime = datetime.datetime.combine(stop_date, stop_time).isoformat(timespec='milliseconds') + 'Z'
                logger.info(f"Executing with {data_source} from {start_datetime} to {stop_datetime}")
                if data_source == "influxdb":
                    results, result_matrix = loader.testing_predict_with_influx(start_datetime, stop_datetime)
                elif data_source == "postgresql":
                    results, result_matrix = loader.testing_predict_with_postgresql(start_datetime, stop_datetime)
            else:
                if not selected_file:
                    st.warning("Please select a local file to test.")
                    return
                logger.info(f"Executing with local file: '{selected_file}'")
                results, result_matrix = loader.testing_predict_with_file(selected_file)
            time_end = time.time()
            time_elapsed = time_end - time_init
            if results is not None:
                display_results(results, result_matrix, time_elapsed, library_name, table_or_graph)
            else:
                st.warning("Execution did not return any results.")
        except Exception as e:
            logger.error(f"❌ Error during execution: {e}")
            st.error(f"Error during execution: {e}")
    

if __name__ == "__main__":
    main()