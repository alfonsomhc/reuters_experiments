"""
print_full_dataframe.py
"""
import pandas as pd

def print_full(x):
    """
    Take a dataframe as input and print all of its rows
    """
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')