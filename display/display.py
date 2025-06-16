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


from testing_tools.testing_so import SOLibraryLoader  # Corrected import path



import os
import datetime
import streamlit as st
import plotly.graph_objects as go
from loguru import logger
import random
import time

lib_dir = "lib"
available_libraries = [f.split(".")[0] for f in os.listdir(lib_dir) if f.endswith(".so")]

# Streamlit app
def main():
    st.title("SO Library Testing Interface")

    
    library_name = st.selectbox(
        "Select a library:",
        available_libraries,
        index=0 if available_libraries else None,
        placeholder="Select a library..."
    )

   
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date(2025, 1, 1))
        start_time = st.time_input("Start Time", datetime.time(0, 0))
    with col2:
        stop_date = st.date_input("Stop Date", datetime.date(2025, 1, 7))
        stop_time = st.time_input("Stop Time", datetime.time(23, 59))
    
    table_or_graph = st.radio("Display Results As:", ("Table", "Graph"), index=1)

    
    if st.button("Execute Testing"):
        if library_name and start_date and stop_date:
            try:
                
                loader = SOLibraryLoader(library_name)
                
                start_datetime = datetime.datetime.combine(start_date, start_time)
                stop_datetime = datetime.datetime.combine(stop_date, stop_time) 

                
                start_date_str = start_datetime.isoformat(timespec='milliseconds') + 'Z'
                stop_date_str = stop_datetime.isoformat(timespec='milliseconds') + 'Z'
                logger.info(f"Start date: {start_date_str}, Stop date: {stop_date_str}")
                
                time_init = time.time()
                
                results, result_matrix = loader.testing(stop_date_str, start_date_str)
                
                time_end = time.time()
                
                time_elapsed = time_end - time_init
                st.markdown(f"<span style='font-size: 24px;'>Time elapsed: {time_elapsed:.2f} seconds for {library_name} for {len(results)} buckets.</span>", unsafe_allow_html=True)
                
                if table_or_graph == "Table":
                    st.markdown("---")
                    create_table(result_matrix)
                else:
                    data = []
                    for bucket, prediction in results.items():
                        if hasattr(prediction, 'columns'):
                            for column in prediction.columns:
                                data.append(go.Scatter(x=prediction.index, y=prediction[column], mode='lines', name=f"{bucket}-{column}"))
                        else:
                            logger.warning(f"Unexpected prediction format for bucket {bucket}: {type(prediction)}")


                    fig = go.Figure(data=data)
                    fig.update_layout(title="Testing Results", xaxis_title="Time", yaxis_title="Value")


                    st.plotly_chart(fig)

            except Exception as e:
                logger.error(f"Error during execution: {e}")
                st.error(f"Error during execution: {e}")
        else:
            st.warning("Please select a library and dates.")

    
    
def create_table(data_dictionary):
    st.markdown("---")
                    
    # Create a table for each bucket
    for bucket, prediction in data_dictionary.items():
        for name, matrix in prediction[-1].items():
            st.markdown(f"### {bucket} - {name}")
            if hasattr(matrix, 'columns'):
                st.dataframe(matrix)
                st.markdown(f"### {bucket} - {name}. NaN table")
                st.dataframe(matrix[matrix.isna().any(axis=1)])
                st.markdown(f"### {bucket} - {name}. dtypes")
                st.dataframe(matrix.dtypes.to_frame(name='Dtype'))
                st.markdown(f"### {bucket} - {name}. Description")
                st.dataframe(matrix.describe())
            else:
                logger.warning(f"Unexpected prediction format for bucket {bucket}: {type(matrix)}")
    
def create_graph(data_dictionary):
    st.markdown("---")
    
    # Create a graph for each bucket
    for bucket, prediction in data_dictionary.items():
        for name, matrix in prediction[-1].items():
            st.markdown(f"### {bucket} - {name}")
            if hasattr(matrix, 'columns'):
                fig = go.Figure()
                for column in matrix.columns:
                    fig.add_trace(go.Scatter(x=matrix.index, y=matrix[column], mode='lines', name=f"{bucket}-{name}-{column}"))
                fig.update_layout(title=f"Graph for {bucket} - {name}", xaxis_title="Time", yaxis_title="Value")
                st.plotly_chart(fig)
            else:
                logger.warning(f"Unexpected prediction format for bucket {bucket}: {type(matrix)}")
if __name__ == "__main__":
    main()