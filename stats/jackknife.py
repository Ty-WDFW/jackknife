import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

def JackKnife(dataframe=None, predictor_column=None, result_column=None, year_column=None):

    '''
    :param dataframe: json dataframe
    :param predictor_column: column that holds the predictor
    :param result_column: column that holds the prediction
    :param year_column: column that holds the years
    :return:
    a dictionary of stats

    takes jsonified dataframe, converts it to a normal dataframe,  runs a jackknife on it
    '''

    if dataframe is None:
        raise ValueError('No dataframe')
    df = pd.read_json(dataframe)
    predictor_mean = df[predictor_column].mean()
    result_mean = df[result_column].mean()

    lm = LinearRegression()

    df_lm = lm.fit(df[predictor_column].values.reshape(-1, 1),
                   df[result_column].values.reshape(-1, 1) )
    #coeeficient of entire dataframe
    coeff = df_lm.coef_[0][0]

    #r2 of entire dataframe
    r2 = r2_score(df[result_column], y_pred=df_lm.predict(df[predictor_column].values.reshape(-1, 1)))

    #empty columns to be filled
    df['b1'] = np.nan
    df['intercept'] = np.nan
    df['r2'] = np.nan

    #run the jackknife
    for index, row in df.iterrows():
        result = df[~(df.index == index)][result_column].values.reshape(-1, 1)
        predictor = df[~(df.index == index)][predictor_column].values.reshape(-1, 1)
        lm_jackknife = lm.fit(predictor, result)
        prediction = lm_jackknife.predict(predictor)
        r2 = r2_score(result, prediction)
        df.at[index, 'r2'] = r2
        df.at[index, 'b1'] = lm_jackknife.coef_[0][0]
        df.at[index, 'intercept'] = lm_jackknife.intercept_[0]

    df['pred_y'] = df['intercept'] + (df['b1'] * df[predictor_column])

    df['re'] = df[result_column] - df['pred_y']
    df['ae'] = abs(df[result_column] - df['pred_y'])
    df['mse'] = df['re'] ** 2
    df['pe'] = df['re'] / df[predictor_column]
    df['ape'] = abs(df['pe'])

    mre = df['re'].mean()
    mae = df['ae'].mean()
    rmse = np.sqrt(df['mse'].mean())
    mpe = df['pe'].mean() * 100
    mape = df['ape'].mean() * 100

    results = {'mre': mre, 'mae': mae,
               'rmse': rmse, 'mpe': mpe,
               'mape': mape, 'dataframe': df}

    return results





