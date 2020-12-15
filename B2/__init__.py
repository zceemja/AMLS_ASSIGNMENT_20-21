"""
Multiclass tasks (cartoon_set dataset)
B2: Eye colour recognition: 5 types of eye colours.
"""
from multiprocessing import Pool

import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

import settings
from Common import common, utils, face_detect
from os import path
import numpy as np

from Common.ui import show_img_grid
from Common.utils import find_best_model

class Model(common.Model):
    def __init__(self, dataset_dir, label_file):
        super(Model, self).__init__("B2", dataset_dir, label_file)
        # self.images = self.images[:300]
        self.images = utils.cache(
            path.join('Data', 'cartoon_without_non_transp_glasses.bin'),
            self._preprocess_data
        )

        self.img_shape = 50


        # self.face_feat = utils.cache(path.join('Data', 'cartoon_face_feat.bin'), face_detect.extract_face_features, images)

        # cachef = path.join('Data', 'cartoon_eyes.bin')
        # eyes = {n: r for n, r in utils.cache(cachef, face_detect.detect_eyes, self.images)}
        mapped_labels = {label[3]: int(label[1]) for label in self.labels[1:]}
        X = np.zeros((len(self.images), self.img_shape, self.img_shape, 3), dtype='uint8')
        Y = np.zeros((len(self.images),), dtype='uint8')
        print("Loading images..", end="")
        for i, img_path in enumerate(self.images):
            # X[i,] = self._load_image(img_path, None)
            img = cv2.imread(img_path)
            h, w = img.shape[0], img.shape[1]
            wc = int(w * 0.1)
            hc = int(h * 0.1)
            img = img[hc:h-hc*2, wc:w//2, ]
            X[i, ] = cv2.resize(img, dsize=(self.img_shape, self.img_shape), interpolation=cv2.INTER_LANCZOS4)
            Y[i] = mapped_labels[path.basename(img_path)]
            print(f"\rLoading images.. {i + 1} of {len(self.images)}", end="")
            # if i > 500:
            #     break
        print(f"\rLoaded all {len(self.images)} images")

        if settings.SHOW_GRAPHS:
            show_img_grid(X, self.img_shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.reshape((len(self.images), -1)), Y, test_size=0.5)
        # self.X_train = self.X_train[0:500, :]
        # self.y_train = self.y_train[0:500]

    def _preprocess_data(self):
        good_images = set()
        images = np.zeros((len(self.images), 50, 50, 3), dtype='uint8')
        print(f'Detecting non-transparent glasses.. 0/{len(self.images)}', end='')
        for i, img_path in enumerate(self.images):
            img = cv2.imread(img_path)
            x, y, w, h = 240, 180, 50, 50
            img = img[x:x + w, y:y + h]
            x, y, w, h = 16, 26, 20, 20

            hsv = cv2.cvtColor(img[x:x+w, y:y+h], cv2.COLOR_RGB2HSV)
            mean = hsv[:, :, 2].mean()
            if settings.SHOW_GRAPHS:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                img = cv2.putText(img, f'{mean:.1f}', (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                colour = (0, 255, 0) if mean > 50 else (0, 0, 255)
                img = cv2.putText(img, f'{mean:.1f}', (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
                images[i, ] = img
            if mean > 50:
                good_images.add(img_path)
            print(f'\rDetecting non-transparent glasses.. {i+1}/{len(self.images)}', end='')
        bad_images_no = len(self.images) - len(good_images)
        print(f'\rDetected non-transparent glasses {bad_images_no}/{len(self.images)} '
                 f'({bad_images_no/len(self.images)*100:.3f}%)')
        if settings.SHOW_GRAPHS:
            show_img_grid(images, (50, 50))
        return good_images

    def _load_image(self, img_path, features=None):
        img = cv2.imread(img_path)  # , cv2.IMREAD_GRAYSCALE)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # if features is not None and len(features) > 0:
        #     x, y, w, h = features[0]
        #     img = img[x:x + w, y:y + h]
        _, gray = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, multichannel=False)

        # for x, y, w, h in features[1]:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
        # for f in features[48:68]:
        #     cv2.drawMarker(img, (f[0], f[1]), color=(0, 255, 0), thickness=3)
        # for f in features[27:36]:
        #     cv2.drawMarker(img, (f[0], f[1]), color=(255, 0, 0), thickness=3)
        # for f in features[0:17]:
        #     cv2.drawMarker(img, (f[0], f[1]), color=(0, 0, 255), thickness=3)
        # for f in features[37:46]:
        #     cv2.drawMarker(img, (f[0], f[1]), color=(255, 0, 255), thickness=3)

        # img = img[250:250 + 30, 190:190 + 30]
        # img = cv2.resize(img, dsize=(8, 8), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(gray.astype('uint8'), cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, dsize=(self.img_shape, self.img_shape), interpolation=cv2.INTER_LANCZOS4)
        # img = cv2.equalizeHist(img)
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

    def __train(self):
        print("Fitting data..")

        # highest_score = 0
        # highest_args = []
        # with Pool(processes=12) as pool:
        #     for (score, args) in pool.imap(self._train, configs):
        #         if score > highest_score:
        #             highest_score = score
        #             highest_args = [args]
        #         elif score == highest_score:
        #             highest_args.append(args)
        #         print(f"Args: {args}, score: {score:.6f}")
        # arg_string = "\n".join([str(a) for a in highest_args])
        # print(f'Score: {highest_score}: \n{arg_string}')
        # print(highest_args)
