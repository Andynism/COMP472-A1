import os
import pandas as pd

dirname = os.path.dirname(__file__)

# Loads a CSV from the datasets folder
def load_csv(filename):
    return pd.read_csv(dirname + f'/../../datasets/{filename}.csv', header=None)

# Writes a dataframe to a CSV file in the output folder
def write_output(filename, dataframe):
    dataframe.to_csv(dirname + f'/../../output/{filename}.csv', header=None)