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
    
    
    total_test = st.number_input("Total Tests", min_value=1, value=5, step=1)
    temporal_windows = st.number_input("Temporal Window (hours)", min_value=1, value=24, step=1)
    
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
               
                results = loader.testing(stop_date_str, start_date_str)

               
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
    
    if st.button("Execute Random Testing"):
        if library_name:
            try:
                # Load the selected library
                loader = SOLibraryLoader(library_name)
                

                year = 2025
          
                for i in range(total_test):
                    month = random.randint(1, 5)
                    if month == 2:
                        day = random.randint(1, 28)  
                    else:
                        day = random.randint(1, 31) 
                    start_date = datetime.date(year, month, day)
        
                    random_hour = random.randint(0, 23)
                    random_time = datetime.time(random_hour, 0, 0)

                    start_datetime = datetime.datetime.combine(start_date, random_time)
                    
                    test_stop_datetime = start_datetime + datetime.timedelta(hours=temporal_windows)
                    
                    logger.info(f"Random Test {i+1}: Start - {start_datetime}, Stop - {test_stop_datetime}")
                    # Format dates to strings
                    test_start_str = start_datetime.isoformat(timespec='milliseconds') + 'Z'
                    test_stop_str = test_stop_datetime.isoformat(timespec='milliseconds') + 'Z'

                    # Execute testing
                    results = loader.testing(test_stop_str, test_start_str)
                    
                    data = []
                    if table_or_graph == "Table":
                        st.subheader(f"Random Test {i+1} Results")
                        st.markdown(f"**Start Time:** {start_datetime} - **Stop Time:** {test_stop_datetime}")
                        st.markdown("---")
                        
                        # Create a table for each bucket
                        for bucket, prediction in results.items():
                            st.markdown(f"### Bucket: {bucket}")
                            if hasattr(prediction, 'columns'):
                                st.dataframe(prediction)
                            else:
                                logger.warning(f"Unexpected prediction format for bucket {bucket}: {type(prediction)}")
                    else:
                        for bucket, prediction in results.items():
                            if hasattr(prediction, 'columns'):
                                for column in prediction.columns:
                                    data.append(go.Scatter(x=prediction.index, y=prediction[column], mode='lines', name=f"{bucket}-{column}"))
                            else:
                                logger.warning(f"Unexpected prediction format for bucket {bucket}: {type(prediction)}")

                        # Create figure
                        fig = go.Figure(data=data)
                        fig.update_layout(title=f"Random Test {i+1}- {start_datetime}, Stop - {test_stop_datetime}: Results", xaxis_title="Time", yaxis_title="Value")

                        # Display graph
                        st.plotly_chart(fig)
                    

            except Exception as e:
                logger.error(f"Error during random execution: {e}")
                st.error(f"Error during random execution: {e}")
        else:
            st.warning("Please select a library.")
    
    

if __name__ == "__main__":
    main()