from pandas import read_csv, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def make_normal_data(dataset):
    keys = dataset.keys()
    for key in keys:
        max_val = dataset[key].max()
        min_val = dataset[key].min()
        dataset[key] = dataset[key].apply(lambda x: (max_val - x) / (max_val - min_val))

    return dataset


def add_new_features(dataset, new_features):
    """

    :param dataset: input dataset
    :param new_features: use this format (('x1', 'x2'), ('x1', 'x3'))
    :return: dataset
    """
    for elem in new_features:
        res = np.ones(len(dataset))
        for key in elem:
            res *= dataset[key]
        dataset["".join(elem)] = res
    return dataset


def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred) / abs(y_true) * 100)


dataset = read_csv('correct_csv/train_data_new.csv', ";")
dataset = dataset.fillna(dataset.median())

target = dataset[["y"]]
train = dataset.drop(["y"], axis=1)
train = make_normal_data(train)
train = add_new_features(train, (('x2', 'x3'), ('x2', 'x4'), ('x2', 'x5')))

models = [
    LinearRegression(copy_X=True, ),  # метод наименьших квадратов
    # RandomForestRegressor(n_estimators=500, max_features='sqrt'),  # случайный лес
    KNeighborsRegressor(n_neighbors=2, algorithm='auto', leaf_size=30, p=2),  # метод ближайших соседей
    # SVR(kernel='linear'),  # метод опорных векторов с линейным ядром
]
test_sizes = [round(0.05 * x, 2) for x in range(1, 20)]

for model in models:
    TestModels = DataFrame()
    tmp = {}
    for test_size in test_sizes:
        tmp['test_size'] = test_size
        Xtrn, Xtest, Ytrn, Ytest = train_test_split(train, target, test_size=test_size, random_state=24)
        m = str(model)
        tmp['Model'] = m[:m.index('(')]
        model.fit(Xtrn, Ytrn)
        y_pred = np.array(model.predict(Xtest))
        tmp['R2_Y'] = r2_score(Ytest, y_pred)
        tmp['error'] = mae(Ytest.values, y_pred)
        TestModels = TestModels.append([tmp])
    # TestModels.set_index('Model', inplace=True)
    TestModels.set_index('test_size', inplace=True)

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)

    TestModels.error.plot(ax=axes, kind='bar', title='Error for %s' % tmp['Model'], grid=True)

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    TestModels.R2_Y.plot(ax=axes, kind='bar', title='R2_Y for %s' % tmp['Model'], grid=True)
    plt.show()
