import sys
import pandas as pd
import numpy as np
import matplotlib
import sklearn
import openpyxl

print("=" * 55)
print("PROJECT 4 - PYTHON ENVIRONMENT TEST")
print("=" * 55)

print("\nPython version:")
print(sys.version)

print("\nInstalled package versions:")
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Scikit-learn:", sklearn.__version__)
print("OpenPyXL:", openpyxl.__version__)

test_data = {
    "customer_id": [1, 2, 3],
    "age": [35, 42, 29],
    "subscribed": ["yes", "no", "yes"]
}

test_df = pd.DataFrame(test_data)

print("\nTest DataFrame:")
print(test_df)

print("\nSUCCESS: Project 4 Python environment is ready.")