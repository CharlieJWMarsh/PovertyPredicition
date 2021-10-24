import pandas as pd
import numpy as np


def normalise_ds(df):
    column_mean = np.sum(df, axis=0) / np.shape(df)[0]
    column_sd = np.sqrt(np.sum((df - column_mean) ** 2, axis=0) / np.shape(df)[0])
    normalised_df = (df - column_mean) / column_sd
    return normalised_df
