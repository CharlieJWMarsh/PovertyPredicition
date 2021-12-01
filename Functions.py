import numpy as np
import heapq


def normalise_ds(df):
    column_mean = np.sum(df, axis=0) / np.shape(df)[0]
    column_sd = np.sqrt(np.sum((df - column_mean) ** 2, axis=0) / np.shape(df)[0])
    normalised_df = (df - column_mean) / column_sd
    np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\safety_normalised_matrix.csv',
               normalised_df, delimiter=',')
    return normalised_df


# Faster way to do this?
def form_ed_matrix(df):
    # Loop through for each column and row
    ed_matrix = np.zeros([np.shape(df)[0], np.shape(df)[0]])
    for i in range(0, np.shape(df)[0]):
        if i % 100 == 1:
            print(i)
        for j in range(0, np.shape(df)[0]):
            ed_matrix[i, j] = np.sqrt(np.sum((df[i, :] - df[j, :]) ** 2))
        ed_matrix[i, i] = 0
    np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\safety_ed_matrix.csv',
               ed_matrix, delimiter=',')
    return ed_matrix


def similarity_scores(df):
    similarity_matrix = np.divide(1, df, out=np.zeros_like(df), where=df != 0)
    np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\safety_similarity_martix.csv',
               similarity_matrix, delimiter=',')
    return similarity_matrix


# Must be faster/smarter way
def threshold_rows(neighbours, df):
    # copy separate arrays so that they are not linked
    df_rows = np.copy(df)
    df_columns = np.copy(df)
    # Loop for each row
    for i in range(0, np.shape(df)[0]):
        if i % 100 == 1:
            print(i)
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
    np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\safety_threshold_martix.csv',
               df_rows, delimiter=',')
    return df_rows


def threshold_rows_2(neighbours, df):
    # copy separate arrays so that they are not linked
    df_rows = np.copy(df)
    df_columns = np.copy(df)
    # Loop for each row
    for i in range(0, np.shape(df)[0]):
        if i % 100 == 1:
            print(i)
        neighbour_rows = df_rows[i, :]
        neighbour_columns = df_columns[:, i]
        largest_x_in_row = heapq.nlargest(neighbours, neighbour_rows)
        x_largest_in_row = heapq.nsmallest(1, largest_x_in_row)
        largest_x_in_column = heapq.nlargest(neighbours, neighbour_columns)
        x_largest_in_column = heapq.nsmallest(1, largest_x_in_column)
        for j in range(0, np.shape(df)[0]):
            if df_rows[i, j] < x_largest_in_row:
                df_rows[i, j] = 0
            if df_columns[j, i] < x_largest_in_column:
                df_columns[j, i] = 0
    for i in range(0, np.shape(df_rows)[0]):
        for j in range(0, np.shape(df_rows)[0]):
            if df_rows[i, j] == 0:
                df_rows[i, j] = df_columns[i, j]
    np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\safety_threshold_martix.csv',
               df_rows, delimiter=',')
    return df_rows


def laplacian(df):
    for i in range(0, np.shape(df)[0]):
        row_sum = sum(df[i, :])
        df[i, :] = -(df[i, :]/row_sum)
        df[i, i] = 1
    np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\safety_laplacian_martix.csv',
               df, delimiter=',')
    return df


def diffusion_map(neighbours, df):
    normalised_ds = normalise_ds(df)
    print("normalised_ds", '\n', normalised_ds, '\n')
    # print("normalised_ds", '\n')

    ed_matrix = form_ed_matrix(normalised_ds)
    print("ed_matrix", '\n', ed_matrix, '\n')
    # print("ed_matrix", '\n')

    similarity_matrix = similarity_scores(ed_matrix)
    print("similarity_matrix", '\n', similarity_matrix, '\n')
    # print("similarity_matrix", '\n')

    threshold_matrix = threshold_rows_2(neighbours, similarity_matrix)
    print("threshold_matrix", '\n', threshold_matrix, '\n')
    # print("threshold_matrix", '\n')

    laplacian_matrix = laplacian(threshold_matrix)
    print("laplacian_matrix", '\n', laplacian_matrix, '\n')
    # print("laplacian_matrix", '\n')

    eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)

    return eigen_values, eigen_vectors
