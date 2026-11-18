import pandas as pd
import numpy as np
import sklearn
import matplotlib
import streamlit

print("=" * 40)
print("FLOWCAST - ENVIRONMENT CHECK")
print("=" * 40)

print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Streamlit:", streamlit.__version__)

print("\nFlowCast environment is ready!")