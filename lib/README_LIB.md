This directory will contain the project's libraries. Once populated, you can use the display with the command `streamlit run display/display.py` to test the libraries.

You can import the library and check the version and information using:

```python
import lib_so_name
e = lib_so_name.PipeExecutor()
e.get_info()
```