"""
Binary task (celeba dataset)
A1: Gender detection: male or female.
"""
import logging
from multiprocessing import Pool

from sklearn import svm, metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

import sys

from tensorflow_estimator.python.estimator.keras import model_to_estimator

import settings

from A1 import adaboost
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from os import path
from sklearn.model_selection import train_test_split, cross_val_score

from Common.face_detect import scale_rect, make_square
from Common import common, face_detect, ui, utils
from Common.ui import show_img_grid
from Common.utils import cache, find_best_model

log = logging.getLogger(__name__)


class Model(common.Model):
    COLOUR_CH = 1

    def __init__(self, dataset_dir, label_file, img_shape=32):
        super(Model, self).__init__("A1", dataset_dir, label_file)
        self.img_shape = img_shape

        faces = utils.cache(path.join('Data', 'celeb_faces.bin'),
                            face_detect.detect_faces, self.images)
        # Labels rows are  [index, img_name, gender, smiling]
        mapped_labels = {label[1]: int(label[2]) for label in self.labels[1:]}
        self.log.info("Preparing images..")

        X = np.zeros((len(faces), img_shape, img_shape, self.COLOUR_CH))
        if self.COLOUR_CH == 1:
            X = X.reshape((len(faces), img_shape, img_shape))
        Y = np.zeros((len(faces), 1))
        for i, (img_path, face) in enumerate(faces):
            x, y, h, w = scale_rect(*make_square(*face), scale=1.0)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img[y:y + h, x:x + w]
            img = cv2.resize(img, dsize=(img_shape, img_shape), interpolation=cv2.INTER_CUBIC)
            img = cv2.equalizeHist(img)
            X[i, ] = img
            gender = mapped_labels[path.basename(img_path)] == 1
            Y[i, 0] = gender
            # Y[i, 0] = 0 if gender else 1
            # Y[i, 1] = 1 if gender else 0
            # if mapped_labels[path.basename(img_path)] == 1:
            #     Y[i, 0] = 1
            # else:
            #     Y[i, 0] = 0
            # X = X.reshape(-1, self.img_shape ** 2)
        X = X/255.
        X = X.reshape((-1, X.shape[1] * X.shape[2] * self.COLOUR_CH))
        # self.pca = PCA(n_components=20*20)
        # X = cache(path.join("Data", "celeb_pca.bin"), self.pca.fit_transform, X)
        # if settings.SHOW_GRAPHS:
            # g = X.reshape((-1, 30, 30))
            # show_img_grid(g, img_shape)

        # X = X.reshape((-1, X.shape[1] * X.shape[2]))
        self.X = X
        self.y = Y.ravel()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        pass
        # self.train_images = self.train_images.T
        # self.train_labels = self.train_labels.T
        # self.test_images = self.test_images.T
        # self.test_labels = self.test_labels.T

    def _train(self, r):
        svc = svm.SVC(kernel='rbf', C=r)
        # abc = AdaBoostClassifier(n_estimators=50, learning_rate=0.7)
        model = svc.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return r, metrics.accuracy_score(self.y_test, y_pred)

    def _train_nn(self):
        model = keras.Sequential([
            keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(self.img_shape, self.img_shape, self.COLOUR_CH)),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu', name='hidden_layer1'),
            keras.layers.Dense(2, activation='sigmoid', name='output')
        ])
        X_train = self.X_train.reshape((-1, self.img_shape, self.img_shape, self.COLOUR_CH))
        X_test = self.X_test.reshape((-1, self.img_shape, self.img_shape, self.COLOUR_CH))
        y_train_hot = utils.binary_to_one_hot(self.y_train)
        y_test_hot = utils.binary_to_one_hot(self.y_test)

        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history = model.fit(
            X_train, y_train_hot,
            steps_per_epoch=100,
            epochs=10,
            validation_data=(X_test, y_test_hot),
            callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
        )
        # if settings.SHOW_GRAPHS:
        #     ui.show_nn_history(history)
        # pred = model.predict(X_test)
        # # pred = np.round(pred)
        # pred = utils.max_arg_one_hot(pred)
        print(f"Accuracy: {model.evaluate(X_test, y_test_hot)[1]:.4f}")

        # find_best_model([
        #     {
        #         'model': [
        #             keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(self.img_shape, self.img_shape, self.COLOUR_CH)),
        #             keras.layers.MaxPooling2D(),
        #             keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        #             keras.layers.MaxPooling2D(),
        #             keras.layers.Flatten(),
        #             keras.layers.Dense(128, activation='relu', name='hidden_layer'),
        #             keras.layers.Dense(2, activation='softmax', name='output')
        #         ],
        #         'fit': {
        #             "epochs": 10,
        #             "steps_per_epoch": 100,
        #             "callbacks":
        #         },
        #         'compile': {
        #             'optimizer': 'adam',
        #             'loss': keras.losses.CategoricalCrossentropy(),
        #             'metrics': ['accuracy']
        #         }
        #     }
        # ], X_train, y_train_hot, 5)
        from tensorflow.python.keras.engine.sequential import Sequential


    def tune_model(self):
        self.log.info("Tuning model")
        return
        self.model =find_best_model(
            [{'model': KNeighborsRegressor, 'n_neighbors': i} for i in range(3, 50, 2)],
            self.X_train, self.y_train, self.KFOLDS
        )
        self.model = find_best_model([
            # {'model': svm.SVC, 'C': 0.1, 'kernel': 'linear'},
            # {'model': svm.SVC, 'C': 1, 'kernel': 'linear'},
            # {'model': svm.SVC, 'C': 10, 'kernel': 'linear'},
            # {'model': svm.SVC, 'C': 0.1, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'C': 1, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'C': 10, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'C': 50, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'C': 100, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'C': 0.1, 'kernel': 'poly', 'degree': 3},
            {'model': svm.SVC, 'C': 1, 'kernel': 'poly', 'degree': 3},
            # {'model': svm.SVC, 'C': 10, 'kernel': 'poly', 'degree': 3},
            # {'model': svm.SVC, 'kernel': 'poly', 'degree': 6},
            # {'model': svm.SVC, 'kernel': 'poly', 'degree': 9},
            # {'model': RandomForestClassifier, 'n_estimators': 100, 'n_jobs': settings.THREADS},
            # {'model': RandomForestClassifier, 'n_estimators': 200, 'n_jobs': settings.THREADS},
            # {'model': RandomForestClassifier, 'n_estimators': 300, 'n_jobs': settings.THREADS},
            # {'model': RandomForestClassifier, 'n_estimators': 400, 'n_jobs': settings.THREADS},
            # {'model': RandomForestClassifier, 'n_estimators': 500, 'n_jobs': settings.THREADS},
            # {'model': KNeighborsRegressor, 'n_neighbors': 3},
            # {'model': KNeighborsRegressor, 'n_neighbors': 8},
            # {'model': KNeighborsRegressor, 'n_neighbors': 12},
        ], self.X_train, self.y_train, self.KFOLDS)

    def train(self):
        self._train_nn()
        return
        # adaboost.train(self.train_images, self.test_images.reshape(-1, self.img_shape**2), self.train_labels, self.test_labels)

        # scores = cross_val_score(svm.SVC(C=0.001), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC C=0.001:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(C=0.01), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC C=0.01:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(C=0.1), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC C=0.1:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(C=1, kernel="linear"), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC C=1:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(C=3, kernel="linear"), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC C=3:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(C=6, kernel="linear"), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC C=6:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(C=10, kernel="linear"), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC C=10:", scores, scores.mean())
        scores = cross_val_score(svm.SVC(C=1, kernel="rbf"), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        print("SVC C=1:", scores, scores.mean())
        scores = cross_val_score(svm.SVC(C=3, kernel="rbf"), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        print("SVC C=3:", scores, scores.mean())
        scores = cross_val_score(svm.SVC(C=6, kernel="rbf"), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        print("SVC C=6:", scores, scores.mean())
        scores = cross_val_score(svm.SVC(C=10, kernel="rbf"), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        print("SVC C=10:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(kernel='poly', degree=3), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC Poly3:", scores, scores.mean())
        scores = cross_val_score(svm.SVC(kernel='poly', degree=4), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        print("SVC Poly4:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(kernel='poly', degree=5), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC Poly5:", scores, scores.mean())
        # scores = cross_val_score(svm.SVC(kernel='poly', degree=8), self.X, self.y, n_jobs=-1, cv=self.KFOLDS)
        # print("SVC Poly8:", scores, scores.mean())
        # scores = cross_val_score(RandomForestClassifier(n_estimators=10, n_jobs=-1), self.X, self.y, cv=self.KFOLDS)
        # print("RF10:", scores, scores.mean())
        # scores = cross_val_score(RandomForestClassifier(n_estimators=50, n_jobs=-1), self.X, self.y, cv=self.KFOLDS)
        # print("RF50:", scores, scores.mean())
        # scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1), self.X, self.y, cv=self.KFOLDS)
        # print("RF100:", scores, scores.mean())

        # scores = cross_val_score(NearestNeighbors(n_neighbors=3, n_jobs=-1), self.X, self.y, cv=self.KFOLDS)
        # print("NN3:", scores, scores.mean())
        # scores = cross_val_score(NearestNeighbors(n_neighbors=5, n_jobs=-1), self.X, self.y, cv=self.KFOLDS)
        # print("NN5:", scores, scores.mean())
        # scores = cross_val_score(NearestNeighbors(n_neighbors=9, n_jobs=-1), self.X, self.y, cv=self.KFOLDS)
        # print("NN9:", scores, scores.mean())

        # param_grid = [
        #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #     # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        # ]
        # svc = svm.SVC()
        # clf = GridSearchCV(svc, param_grid)
        # clf.fit(self.train_images.reshape(-1, self.img_shape**2), self.train_labels)
        # svc = SVC(probability=True, kernel='linear')
        # Create adaboost classifer object
        # with Pool() as pool:
        #     for (r, p) in pool.imap(self._train, np.logspace(1, 4, 16)):
        #         print(f"Accuracy with r={r}:", p)
        #     abc = AdaBoostClassifier(n_estimators=50, learning_rate=r)
        # # abc = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1)
        # # Train Adaboost Classifer
        #     model = abc.fit(self.X_train, self.y_train)
        #
        # # Predict the response for test dataset
        #     y_pred = model.predict(self.X_test)
        #
        # # Model Accuracy, how often is the classifier correct?
        #     print(f"Accuracy with r={r}:", metrics.accuracy_score(self.y_test, y_pred))
        #


        # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        # # model = tf.keras.Sequential([
        # #     tf.keras.layers.Flatten(input_shape=(self.img_shape, self.img_shape)),
        # #     tf.keras.layers.Dense(1600, activation='relu'),
        # #     tf.keras.layers.Dense(1600, activation='relu'),
        # #     tf.keras.layers.Dense(2, activation='softmax')
        # # ])
        # model = tf.keras.Sequential([
        #     tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(self.img_shape, self.img_shape, 3)),
        #     # tf.keras.layers.Flatten(input_shape=(self.img_shape, self.img_shape)),
        #     tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(2)
        # ])
        # model.compile(optimizer='adam',
        #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #               metrics=['accuracy'])
        # epochs = 10
        # try:
        #     history = model.fit(self.train_images, self.train_labels, epochs=epochs)
        #
        #     epochs_range = range(epochs)
        #     acc = history.history['accuracy']
        #     # val_acc = history.history['val_accuracy']
        #     loss = history.history['loss']
        #     # val_loss = history.history['val_loss']
        #     plt.figure(figsize=(8, 8))
        #     plt.subplot(1, 2, 1)
        #     plt.plot(epochs_range, acc, label='Training Accuracy')
        #     # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        #     plt.legend(loc='lower right')
        #     plt.title('Training and Validation Accuracy')
        #     plt.subplot(1, 2, 2)
        #     plt.plot(epochs_range, loss, label='Training Loss')
        #     # plt.plot(epochs_range, val_loss, label='Validation Loss')
        #     plt.legend(loc='upper right')
        #     plt.title('Training and Validation Loss')
        #     plt.show()
        #
        # except KeyboardInterrupt:
        #     pass
        # test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=2)
        # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # pass
