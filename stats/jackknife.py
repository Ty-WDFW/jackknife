import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

matplotlib.use('Agg')
sns.set_style('darkgrid')

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

    df[year_column] = pd.to_datetime(df[year_column], format='%Y')
    df.sort_values(year_column, inplace=True)
    print(df)

    predictor_mean = df[predictor_column].mean()
    result_mean = df[result_column].mean()

    lm = LinearRegression()

    df_lm = lm.fit(df[predictor_column].values.reshape(-1, 1),
                   df[result_column].values.reshape(-1, 1))

    # coeeficient of entire dataframe
    coeff = df_lm.coef_[0][0]

    # r2 of entire dataframe
    r2 = r2_score(df[result_column], y_pred=df_lm.predict(df[predictor_column].values.reshape(-1, 1)))

    # empty columns to be filled
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

    #plot this bad boy


    img = io.BytesIO()
    plot_df = df.set_index(year_column)
    fig, ax = plt.subplots()
    ax.plot(plot_df[result_column], '--.', label='Observed')
    ax.plot(plot_df['pred_y'], '--.', label='Estimated')
    ax.set_ylabel(result_column)
    ax.set_xlabel(year_column)
    ax.legend()
    fig.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    results = {'mre': mre, 'mae': mae,
               'rmse': rmse, 'mpe': mpe,
               'mape': mape, 'dataframe': df.reset_index(drop=True),
               'graph': 'data:image/png;base64,{}'.format(graph_url)}

    return results





