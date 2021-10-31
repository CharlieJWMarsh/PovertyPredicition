import pandas as pd
import numpy as np


def normalise_ds(df):
    column_mean = np.sum(df, axis=0) / np.shape(df)[0]
    column_sd = np.sqrt(np.sum((df - column_mean) ** 2, axis=0) / np.shape(df)[0])
    normalised_df = (df - column_mean) / column_sd
    return normalised_df


# Faster way to do this?
def form_ed_matrix(df):
    # Loop through for each column and row
    ed_matrix = np.zeros([np.shape(df)[0], np.shape(df)[0]])
    for i in range(0, np.shape(df)[0]):
        for j in range(0, np.shape(df)[0]):
            ed_matrix[i, j] = np.sqrt(np.sum((df[i, :] - df[j, :]) ** 2))
    return ed_matrix


def similarity_scores(df):
    similarity_matrix = np.divide(1, df, out=np.zeros_like(df), where=df != 0)
    return similarity_matrix


# Does the symmetry along leading diagonal matter? x2 on threshold
# Must be faster/smarter way
def threshold(neighbours, df):
    # Loop for each row
    for i in range(0, np.shape(df)[0]):
        neighbour_scores = df[i, :]
        for j in range(0, neighbours):
            # Remove the top stated number of rows
            neighbour_scores = np.delete(neighbour_scores, np.where(neighbour_scores == max(neighbour_scores)))
        matrix_row = df[i, :]
        # Set everything in the row below the score to 0
        df[i, :] = [0 if matrix_row <= max(neighbour_scores) else matrix_row for matrix_row in matrix_row]
    return df


def laplacian(df):
    for i in range(0, np.shape(df)[0]):
        row_sum = sum(df[i, :])
        df[i, :] = -(df[i, :]/row_sum)
        df[i, i] = 1
    return df
