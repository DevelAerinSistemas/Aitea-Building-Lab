# AItea Building Lab

## Installation and initialization

1. Make sure your machine has the following modules and software correctly installed:
    1. Running Debian 12.0, check using command: `lsb_release -a`
    2. Python 3.11, if not install it with the following commands:
        ```bash
        sudo apt update
        sudo apt install zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget liblzma-dev
        sudo apt install python3.11
        ```
    3. Available connection to InfluxDB
    4. Nuitka library installed, version '2.7.8' on Python 3.11 (flavor 'Debian Python') commercial grade '3.8.4'
    5. C Compiler, if not use: `sudo apt install build-essential`
    6. Python development headers, if not use: `sudo apt install python3.11-dev`
    7. C cache installed for quick development, if not use: `sudo apt install ccache`

2. Python environment
    1. Create a virtual python environment with `python3 -m venv {environment_path}` (substitute `{environment_path}` by your actual path)
    2. Clone the repository using `git clone --recurse-submodules {repository_url}` (substitute `{repository_url}` by current **AItea Building Lab** repository, normally would be `https://github.com/DevelAerinSistemas/Aitea-Building-Lab`). The flag `--recurse-submodules` is to include automatically submodules.
    3. Modify the **.env** file variable `PYTHON_ENV` with the path of the virtual python environment just created
    4. Activate your python environment using `source {environment_path}/bin/activate`
    5. Install python dependencies from **/config/requirements.txt** using `pip3 install -r /config/requirements.txt` (executed from project root path)

3. Setting up
    1. The configuration file used by default is **config/global.json**. If needed something different, make your own version and modify the `CONFIG_PATH` variable in **.env** file.
    2. Modify as well `CONNECTIONS_PATH`variable in **/aitea_connectors/.env** to point to the global configuration file of **AItea Building Lab** you included in the `CONFIG_PATH` variable in the last step.
    3. Modify content for keys **influxdb** and **postresql** in the configuration JSON to fit your InfluxDB and PostgreSQL connections and configurations.
    4. Read carefully [project's wiki](https://github.com/DevelAerinSistemas/Aitea-Building-Lab/wiki) to understand how to develop a transformation/model (stored at **models_warehouse/**) and include it in the pipeline execution (in file **pipes_schedules/pipe_plan.json**).

4. Testing everything works with prepared test
    1. Familiarize yourself with code from *DemoTransform* and *DemoModel* classes from **models_warehouse/demo.py** 
    2. Familiarize yourself with *demo* pipeline from **pipes_schedules/pipe_plan_demo.json**
    3. Run the global test, which generates testing data in your InfluxDB database and uses it to generate your dummy model *.pkl* and library *.so* (both stored at **lib/**) using `python3 -m testing_tools.testing_demo`
    4. Run the following command to test library generation and use GUI implementation: `streamlit run display/display.py`

5. Using the application
    1. Create your own transformations and models, storing them at **/models_warehouse**. 
    2. Include a new pipeline with your transformations and models and the required data they need to run into file **pipes_schedules/pipe_plan.json**
    3. Run the application test to generate your model *.pkl* and library *.so* (both stored at **lib/**) using `python3 -m testing_tools.testing_app`
    4. Run the following command to test library generation and use GUI implementation: `streamlit run display/display.py`


