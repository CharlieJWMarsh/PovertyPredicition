import pandas as pd
import numpy as np


def normalise_ds(df):
    column_mean = np.sum(df, axis=0) / np.shape(df)[0]
    column_sd = np.sqrt(np.sum((df - column_mean) ** 2, axis=0) / np.shape(df)[0])
    normalised_df = (df - column_mean) / column_sd
    return normalised_df


def form_ed_matrix(df):
    ed_matrix = np.zeros([np.shape(df)[0], np.shape(df)[0]])
    for i in range(0, np.shape(df)[0]):
        for j in range(0, np.shape(df)[0]):
            ed_matrix[i, j] = np.sqrt(np.sum((df[i, :] - df[j, :]) ** 2))
    return ed_matrix


def similarity_scores(df):
    similarity_matrix = np.divide(1, df, out=np.zeros_like(df), where=df != 0)
    return similarity_matrix
