from multiprocessing import Pool

from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from numpy import linspace


def __test_model(data):
    train_x, test_x, train_y, test_y, n_est, learn_rate = data
    svc = SVC(probability=True, kernel='linear')
    abc = AdaBoostClassifier(n_estimators=n_est, base_estimator=svc, learning_rate=learn_rate)
    model = abc.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    return metrics.accuracy_score(test_y, y_pred), n_est, learn_rate


def train(X_train, X_test, y_train, y_test):
    params = []
    for n in linspace(5, 100, 4):
        for lr in linspace(0.1, 2, 4):
            params.append((X_train, X_test, y_train, y_test, int(n), float(lr)))

    pool = Pool()
    results = pool.imap(__test_model, params)
    for result in results:
        acc, n_est, lear_r = result
        print(f"Model N_EST={n_est} Learning Rate={lear_r} Accuracy={acc}")
    pool.close()
