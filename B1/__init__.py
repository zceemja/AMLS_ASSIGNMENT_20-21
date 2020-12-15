"""
Multiclass tasks (cartoon_set dataset)
B1: Face shape recognition: 5 types of face shapes
"""
from os import path

import cv2
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

import settings
from Common import common
from Common.pso import Pso
from Common.ui import show_img_grid

from Common.utils import find_best_model


class Model(common.Model):
    def __init__(self, dataset_dir, label_file):
        super(Model, self).__init__("B1", dataset_dir, label_file)
        # saved_file = path.join('B1', 'saved.npz')
        self.img_shape = 50

        mapped_labels = {label[3]: int(label[2]) for label in self.labels[1:]}
        X = np.zeros((len(self.images), self.img_shape, self.img_shape), dtype='uint8')
        Y = np.zeros((len(self.images),), dtype='uint8')
        print("Loading images..", end="")
        for i, img_path in enumerate(self.images):
            X[i, :, :] = self._load_image(img_path)
            Y[i] = mapped_labels[path.basename(img_path)]
            print(f"\rLoading images.. {i+1} of {len(self.images)}", end="")
        print()

        if settings.SHOW_GRAPHS:
            show_img_grid(X, self.img_shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.reshape((len(self.images), -1)), Y)

    def _load_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img = img[180:180 + 250, 125:125 + 250]
        img = cv2.resize(img, dsize=(self.img_shape, self.img_shape), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.equalizeHist(img)
        return img

    def tune_model(self):
        self.model = find_best_model([
            {'model': svm.SVC, 'C': 0.1, 'kernel': 'linear'},
            {'model': svm.SVC, 'C': 1, 'kernel': 'linear'},
            {'model': svm.SVC, 'C': 10, 'kernel': 'linear'},
            {'model': svm.SVC, 'C': 0.1, 'kernel': 'rbf'},
            {'model': svm.SVC, 'C': 1, 'kernel': 'rbf'},
            {'model': svm.SVC, 'C': 10, 'kernel': 'rbf'},
            {'model': svm.SVC, 'C': 0.1, 'kernel': 'poly', 'degree': 3},
            {'model': svm.SVC, 'C': 1, 'kernel': 'poly', 'degree': 3},
            {'model': svm.SVC, 'C': 10, 'kernel': 'poly', 'degree': 3},
            {'model': svm.SVC, 'kernel': 'poly', 'degree': 6},
            {'model': svm.SVC, 'kernel': 'poly', 'degree': 9},
            {'model': RandomForestClassifier, 'n_estimators': 100, 'n_jobs': settings.THREADS},
            {'model': RandomForestClassifier, 'n_estimators': 200, 'n_jobs': settings.THREADS},
            {'model': RandomForestClassifier, 'n_estimators': 300, 'n_jobs': settings.THREADS},
            {'model': RandomForestClassifier, 'n_estimators': 400, 'n_jobs': settings.THREADS},
            {'model': RandomForestClassifier, 'n_estimators': 500, 'n_jobs': settings.THREADS},
        ], self.X_train, self.y_train, self.KFOLDS)

    def _train(self, args):
        alpha = args[0]
        degree = args[1]

        svc = svm.SVC(C=alpha, degree=degree, kernel='poly')
        svc.fit(self.X_train, self.y_train)
        predicted = svc.predict(self.X_test)
        score = accuracy_score(self.y_test, predicted)
        print(f'current alpha: {alpha}, degree={degree}, val: {score}')

        # model = MLPClassifier(alpha=alpha, max_iter=1000)
        # model.fit(self.X_train, self.y_train)
        # predicted = model.predict(self.X_test)
        # return accuracy_score(self.y_test, predicted), alpha

        return score

    def __train(self):
        print("Fitting data..")

        # pso = Pso(swarmsize=4, maxiter=14)
        #
        # bp, value = pso.run(
        #     self._train,
        #     np.array([1, 1]),
        #     np.array([1000, 50])
        # )
        #
        # v = self._train(bp)
        # print('Test loss:', bp)
        # print('Test accuracy:', value, v)

        # param_grid = [
        #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #     {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        # ]
        # svc = svm.SVC()
        # clf = GridSearchCV(svc, param_grid)
        # clf.fit(self.X_train, self.y_train)


        # with Pool() as pool:
        #     for (score, alpha) in pool.imap(self._train, [0.001, 0.01, 0.1, 1]):
        #         print(f"Alpha: {alpha}, score: {score:.6f}")

        # print("Testing data..")
        # predicted = clf.predict(self.X_test)
        # print(f'Accuracy {accuracy_score(self.y_test, predicted):.4f}')
