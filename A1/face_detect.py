import cv2
import matplotlib.pyplot as plt
import requests
import logging
import os
import common
from multiprocessing import Pool


log = logging.getLogger(__name__)

HAAR_CASCADE_URLS = {
    'face': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
    'eyes': 'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml',
    'smile': 'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml',
}


def download_classifiers():
    for name, url in HAAR_CASCADE_URLS.items():
        if os.path.isfile(name + '.xml'):
            continue
        fpath = os.path.abspath(name + '.xml')
        log.info(f"Download haar cascade for {name} to {fpath}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fpath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)

download_classifiers()
face_cascade = cv2.CascadeClassifier('face.xml')


def __get_face(fpath):
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    if len(faces) == 1:
        return faces[0]
    return None


def detect_faces(images):
    dsum = 0
    pool = Pool()
    detects = pool.imap(__get_face, images)
    for i, d in enumerate(detects):
        if d is not None:
            dsum += 1
        print(f'\rDetecting faces {i}/{len(images)}', end='')
    pool.close()
    print('\r', end='')

    # for fpath in images:
    #     img = cv2.imread(fpath)
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    #     if len(faces) == 1:
    #         detected += 1
    #         continue
    #     if not show_images:
    #         continue
    #     for x, y, w, h in faces:
    #         cv2.rectangle(img, (x, y), (w+x, h+y), color=(0, 255, 0), thickness=5)
    #     cv2.imshow('Face Detect', img)
    #     log.info(f"Loaded image {fpath} with {len(faces)} detected faces")
    #     k = cv2.waitKey(0)
    #     if k == 27:  # Esc key
    #         break
    log.info(f"Total detected faces {dsum} of {len(images)} ({dsum/len(images)*100:.2f}%)")
