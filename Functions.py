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


# Must be faster/smarter way
def threshold_rows(neighbours, df):
    # copy separate arrays so that they are not linked
    df_rows = np.copy(df)
    df_columns = np.copy(df)
    # Loop for each row
    for i in range(0, np.shape(df)[0]):
        neighbour_rows = df_rows[i, :]
        neighbour_columns = df_columns[:, i]
        for j in range(0, neighbours):
            # Remove the top stated number of rows
            neighbour_rows = np.delete(neighbour_rows, np.where(neighbour_rows == max(neighbour_rows)))
            neighbour_columns = np.delete(neighbour_columns, np.where(neighbour_columns == max(neighbour_columns)))
        matrix_row = df_rows[i, :]
        matrix_column = df_columns[:, i]
        # Set everything in the row below the score to 0
        df_rows[i, :] = [0 if value <= max(neighbour_rows) else value for value in matrix_row]
        df_columns[:, i] = [0 if value <= max(neighbour_columns) else value for value in matrix_column]
    for i in range(0, np.shape(df_rows)[0]):
        for j in range(0, np.shape(df_rows)[0]):
            if df_rows[i, j] == 0:
                df_rows[i, j] = df_columns[i, j]
    return df_rows


def threshold_columns(neighbours, df):
    # Loop for each row
    print(df)
    for i in range(0, np.shape(df)[1]):
        neighbour_columns = df[:, i]
        for j in range(0, neighbours):
            # Remove the top stated number of rows
            neighbour_columns = np.delete(neighbour_columns, np.where(neighbour_columns == max(neighbour_columns)))
        matrix_column = df[:, i]
        # Set everything in the row below the score to 0
        df[:, i] = [0 if matrix_column <= max(neighbour_columns) else matrix_column for matrix_column in matrix_column]
    return df


def laplacian(df):
    for i in range(0, np.shape(df)[0]):
        row_sum = sum(df[i, :])
        df[i, :] = -(df[i, :]/row_sum)
        df[i, i] = 1
    return df
