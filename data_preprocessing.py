import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    data.replace('?', np.nan, inplace=True)
    data = data.astype(float)


    col_mean = data.mean()
    data.fillna(col_mean, inplace=True)

    return data

def split_data(data):

    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test
