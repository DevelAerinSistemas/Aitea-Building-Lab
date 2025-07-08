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
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import base64
import subprocess
from utils.logger_config import get_logger
logger = get_logger()

@logger.catch
def create_so(
    model_path: str, 
    template_path: str = "utils/so_template.py"
) -> None:
    """
    Creates a shared object (SO) file from a serialized machine learning pipeline.
    Args:
        model_path (str): Path to the serialized model file (e.g., '.pkl').
        template_path (str, optional): Path to the template Python file for the executor. Defaults to "templates/so_template.py".
    """
    from dotenv import load_dotenv
    import os
    load_dotenv()
    from utils.file_utils import load_json_file
    python_env = os.getenv("PYTHON_ENV")
    global_config = load_json_file(os.getenv("CONFIG_PATH"))
    models_path = global_config.get("models_path")
    libs_path = global_config.get("libs_path")

    with open(model_path, "rb") as f:
        pipe_data = f.read()
        pipe_data_base64 = base64.b64encode(pipe_data).decode("utf-8")
    with open(template_path, "r") as f:
        template_code = f.read()
    executor_code = template_code.replace("{pipe_data_base64}", pipe_data_base64)
    name_so = model_path.split("/")[1].split(".")[0]
    name_py = name_so + ".py"
    
    with open(name_py, "w") as f:
        f.write(executor_code)

    command = [
        f"{python_env}/bin/nuitka",
        "--module", name_py,
        f"--include-package={models_path}",
        "--include-package=metaclass",
        "--include-package=utils",
        "--show-modules",
        f"--output-dir={libs_path}",
        "--remove-output"
    ]
    try:
        logger.info(f"⚙️ Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.success("✅ Command executed successfully")
        if result.stdout:
            logger.info(f"💬 Standard Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"⚠️ Standard Error Informational/Verbose Output: {result.stderr}")
    except subprocess.CalledProcessError as err:
        logger.error(f"❌ Command failed with exit code {err.returncode}")
        logger.error(f"❌ Command: {' '.join(err.cmd)}")
        if err.stdout:
            logger.error(f"❌ Standard Output (from failed command):\n{err.stdout}")
        if err.stderr:
            logger.error(f"❌ Standard Error (from failed command):\n{err.stderr}")
    finally:
        # Clean up the temporary file
        import os
        import glob
        if os.path.exists(name_py):
            os.remove(name_py)
            files = glob.glob("lib/*.pyi")
            for file in files:
                os.remove(file)
            logger.info(f"💬 Temporary file '{file}' removed.")


if __name__ == "__main__":
    # Example usage
    create_so(model_path = "training_models/consumption_analysis.pkl")
    logger.success("✅ SO file created successfully.")
