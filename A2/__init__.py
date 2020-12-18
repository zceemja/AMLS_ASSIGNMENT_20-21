"""
Binary task (celeba dataset)
A2: Emotion detection: smiling or not smiling.
"""
from multiprocessing import Pool

from sklearn import svm, metrics
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

import settings
from Common import common
from Common import face_detect
import numpy as np
import cv2
from os import path, listdir

from Common.face_detect import detect_faces
from Common import utils, ui
from matplotlib import pyplot as plt

from Common.utils import find_best_model


class Model(common.Model):
    PSIZE = 100  # preview image resolution

    def __init__(self, dataset_dir, label_file):
        super(Model, self).__init__("A2", dataset_dir, label_file)

    def prepare_data(self, images_dir, labels_file, train=True):
        images = [path.join(images_dir, f) for f in listdir(images_dir)]
        mapped_labels = self.map_labels(self.read_csv(labels_file))

        if train:
            faces = utils.cache(path.join(settings.DATA_PATH, 'celeb_faces.bin'), face_detect.detect_faces, images)
        else:
            faces = face_detect.detect_faces(images)
        for i, (fpath, rect) in enumerate(faces):
            faces[i] = (fpath, face_detect.scale_rect(*rect, scale=1.2))
        if train:
            face_feat = utils.cache(path.join(settings.DATA_PATH, 'celeb_face_feat.bin'),
                                    face_detect.extract_face_features, faces)
        else:
            face_feat = face_detect.extract_face_features(faces)

        preview_img = np.zeros((len(faces), self.PSIZE, self.PSIZE, 3), dtype='uint8')
        X = np.zeros((len(faces), 68, 2))
        Y = np.zeros((len(faces)), dtype=bool)

        for i, (img_path, face) in enumerate(faces):
            features = face_feat[i]

            # normalising
            f = features.astype('float32')
            # f -= np.amin(f, axis=0)
            # f /= np.amax(f, axis=0)
            X[i, ] = f  #[48:68]

            Y[i] = 1 if mapped_labels[path.basename(img_path)] == 1 else 0
            x, y, h, w = face_detect.scale_rect(*face_detect.make_square(*face), scale=1.3)
            img = cv2.imread(img_path)
            # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            crop_x = features[3, 0]
            crop_y = features[30, 1]
            crop_h = features[13, 0] - crop_x
            crop_w = features[8, 1] - crop_y
            crop_x, crop_y, crop_h, crop_w = face_detect.make_square(crop_x, crop_y, crop_h, crop_w)
            # gray = gray[crop_y:crop_y + crop_w, crop_x:crop_x + crop_h]
            # gray = gray[y:y + h, x:x + w]
            # gray = cv2.resize(gray, dsize=(25, 25), interpolation=cv2.INTER_CUBIC)
            # gray = cv2.equalizeHist(gray)

            # Lips
            for f in features[48:68]:
                cv2.circle(img, (f[0], f[1]), 1, color=(0, 255, 0), thickness=2)
            # Eyebrows
            for f in features[17:27]:
                cv2.circle(img, (f[0], f[1]), 1, color=(155, 155, 0), thickness=2)
            # Eyes
            for f in features[36:48]:
                cv2.circle(img, (f[0], f[1]), 1, color=(0, 155, 155), thickness=2)
            # Nose
            for f in features[27:36]:
                cv2.circle(img, (f[0], f[1]), 1, color=(255, 0, 0), thickness=2)
            # Face shape
            for f in features[0:17]:
                cv2.circle(img, (f[0], f[1]), 1, color=(0, 0, 255), thickness=2)

            # X[i, ] = features[49:69].flatten()
            # X[i, ] = gray
            preview_img[i, ] = cv2.resize(img[y:y + h, x:x + w], dsize=(self.PSIZE, self.PSIZE), interpolation=cv2.INTER_CUBIC)
            #
            # img = img[y:y + h, x:x + w]
            # img = cv2.resize(img, dsize=(30, 30), interpolation=cv2.INTER_CUBIC)
            # img = img[img_h:img_h * 2, :]
            # img = cv2.equalizeHist(img)
            # rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # _, img = hog(img, orientations=4, pixels_per_cell=(4, 4),  cells_per_block=(1, 1), visualize=True, multichannel=False)

            # write RGB
            # Ximg[i, ] = img

        # X = X.reshape(-1, self.img_shape ** 2)
        if settings.SHOW_GRAPHS:
            ui.show_img_grid(preview_img, (self.PSIZE, self.PSIZE))

        # Normalise X
        X = X.reshape((-1, X.shape[1]*X.shape[2]))
        return X, Y

    def map_labels(self, rows):
        generator = iter(rows)  # make sure it's a generator
        next(generator)  # skip header
        # Labels rows are  [index, img_name, gender, smiling]
        return {label[1]: int(label[3]) for label in generator}

    def _train(self, args):
        svc = svm.SVC(kernel='poly', C=0.02154, degree=args)
        # abc = AdaBoostClassifier(n_estimators=50, learning_rate=0.7)
        model = svc.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return args, metrics.accuracy_score(self.y_test, y_pred)

    def tune_model(self):

        # C_range = np.logspace(-2, 10, 15)
        # gamma_range = np.logspace(-9, 3, 15)
        # param_grid = dict(gamma=gamma_range, C=C_range)
        # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=settings.THREADS)
        # grid.fit(self.X_train, self.y_train)
        # print("The best parameters are %s with a score of %0.2f"
        #       % (grid.best_params_, grid.best_score_))

        self.model = find_best_model([
            # {'model': svm.SVC, 'C': 0.001, 'kernel': 'linear'},
            {'model': svm.SVC, 'C': 0.01, 'kernel': 'linear'},
            {'model': svm.SVC, 'C': 0.1, 'kernel': 'linear'},
            {'model': svm.SVC, 'C': 1, 'kernel': 'linear'},
            {'model': svm.SVC, 'C': 10, 'kernel': 'linear'},
            # {'model': svm.SVC, 'C': 0.01, 'kernel': 'rbf'},
            {'model': svm.SVC, 'C': 0.1, 'kernel': 'rbf'},
            {'model': svm.SVC, 'C': 1, 'kernel': 'rbf'},
            {'model': svm.SVC, 'C': 10, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'C': 100, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'C': 1000, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'gamma': 0.001, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'gamma': 0.01, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'gamma': 0.1, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'gamma': 1, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'gamma': 10, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'gamma': 100, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'gamma': 1000, 'kernel': 'rbf'},
            # {'model': svm.SVC, 'gamma': 0.01, 'C': 10000, 'kernel': 'rbf'},
            {'model': svm.SVC, 'kernel': 'poly', 'C': 0.1},
            {'model': svm.SVC, 'kernel': 'poly', 'C': 1},
            {'model': svm.SVC, 'kernel': 'poly', 'C': 10},
            {'model': svm.SVC, 'kernel': 'poly', 'degree': 3},
            {'model': svm.SVC, 'kernel': 'poly', 'degree': 6},
            {'model': svm.SVC, 'kernel': 'poly', 'degree': 9},
            {'model': RandomForestClassifier, 'n_estimators': 100, 'n_jobs': settings.THREADS},
            {'model': RandomForestClassifier, 'n_estimators': 200, 'n_jobs': settings.THREADS},
            {'model': RandomForestClassifier, 'n_estimators': 300, 'n_jobs': settings.THREADS},
            {'model': RandomForestClassifier, 'n_estimators': 400, 'n_jobs': settings.THREADS},
            {'model': RandomForestClassifier, 'n_estimators': 500, 'n_jobs': settings.THREADS},
        ], self.X_train, self.y_train, self.KFOLDS)

    def _train(self):
        # model = keras.Sequential([
        #     keras.layers.Flatten(input_shape=(30, 30)),
        #     keras.layers.Dense(128, activation='relu'),
        #     keras.layers.Dense(1)
        # ])
        # model.compile(optimizer='adam',
        #               # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #               loss='binary_crossentropy',
        #               metrics=['accuracy'])
        # epochs = 100
        # history = model.fit(self.X_train, self.y_train, epochs=epochs)
        # pred = model.predict(self.X_test)
        # print(f"Accuracy: {metrics.accuracy_score(self.y_test, pred):.4f}")

        model = keras.Sequential([
            keras.layers.Conv2D(8, 3, padding='same', activation='relu', input_shape=(50, 50, 3)),
            keras.layers.MaxPooling2D(),
            keras.layers.LeakyReLU(alpha=0.3),
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.LeakyReLU(alpha=0.3),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu', name='hidden_layer'),
            keras.layers.Dense(1, activation='sigmoid', name='output')
        ])

        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        history = model.fit(
            self.X_train, self.y_train,
            steps_per_epoch=100,
            epochs=150,
            validation_data=(self.X_test, self.y_test)
        )

        ui.show_nn_history(history)

        pred = model.predict(self.X_test)
        pred = np.around(pred).astype(bool).reshape((-1, ))
        print(f"Accuracy: {metrics.accuracy_score(self.y_test, pred):.4f}")

        # with Pool() as pool:
        #     for (r, p) in pool.imap(self._train, range(3, 10)):
        #         print(f"Accuracy with r={r:.5f}: {p:.4f}")

        # adaboost.train(self.train_images.reshape(-1, 50*25), self.test_images.reshape(-1, 50*25), self.train_labels, self.test_labels)
        # param_grid = [
        #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #     # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        # ]
        # svc = svm.SVC()
        # # clf = GridSearchCV(svc, param_grid)
        # clf.fit(self.train_images.reshape(-1, 40*20), self.train_labels)
        # # with Pool() as pool:
        # #     for (score, alpha) in pool.imap(self._train, [0.001, 0.01, 0.1, 1]):
        # #         print(f"Alpha: {alpha}, score: {score:.6f}")
        #
        # print("Testing data..")
        # predicted = clf.predict(self.test_images.reshape(-1, 40*20))
        # print(f'Accuracy {accuracy_score(self.test_labels, predicted):.4f}')
