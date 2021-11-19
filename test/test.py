# this import requires python 3.5 +
import importlib.util

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("utilities", "../src/utilities.py")
    utilities = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utilities)
    utilities.test_utilities()
