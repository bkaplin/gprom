from pandas import read_csv, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from itertools import combinations


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
    :param new_features: use this format (('x1', 'x2'), ('x1', 'x3', 'x5'), ...)
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
G_train = make_normal_data(train)

keys = train.keys()
main_combinatoins_list = []
models = [
    LinearRegression(copy_X=True, ),  # метод наименьших квадратов
    # RandomForestRegressor(n_estimators=500, max_features='sqrt'),  # случайный лес
    KNeighborsRegressor(n_neighbors=2, algorithm='auto', leaf_size=30, p=2),  # метод ближайших соседей
    # SVR(kernel='poly'),  # метод опорных векторов с линейным ядром
]

test_sizes = [round(0.05 * x, 2) for x in range(1, 20)]
MIN_ERROR = 100
# цикл по различным комбинациям дополнительных параметров
for i in range(2, len(keys)):
    print(f"MAIN CYCLE COUNTER: {i}")
    main_combinatoins_list += combinations(keys, i)
    index = 0
    for j in range(1, len(main_combinatoins_list)):
        for k in range(j, len(main_combinatoins_list)):
            train = add_new_features(G_train.copy(), main_combinatoins_list[k-j:k])

            # для каждой модели из списка
            for model in models:
                TestModels = DataFrame()
                tmp = {}
                for test_size in test_sizes:
                    tmp['test_size'] = test_size
                    m = str(model)

                    Xtrn, Xtest, Ytrn, Ytest = train_test_split(train, target, test_size=test_size, random_state=24)
                    tmp['Model'] = m[:m.index('(')]
                    model.fit(Xtrn, Ytrn)

                    y_pred = np.array(model.predict(Xtest))

                    tmp['R2_Y'] = r2_score(Ytest, y_pred)
                    tmp['error'] = mae(Ytest.values, y_pred)

                    if tmp['error'] < MIN_ERROR:
                        MIN_ERROR = tmp['error']
                        print(f"MODEL: {tmp['Model']}, TEST_SIZE: {test_size}, FEATURES: {train.keys()}")

                    TestModels = TestModels.append([tmp])
            # делаем индекс по названию модели
            # TestModels.set_index('Model', inplace=True)
                TestModels.set_index('test_size', inplace=True)
