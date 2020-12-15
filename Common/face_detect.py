import cv2
import dlib
import requests
import logging
import os
from multiprocessing import Pool
import numpy as np

log = logging.getLogger(__name__)

HAAR_BASE_DIR = "/usr/share/opencv4/haarcascades"
HAAR_BASE_DIR_USER = "Data"
HAAR_GITHUB_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
HAAR_CASCADE_URLS = {
    'face': HAAR_BASE_DIR + 'haarcascade_frontalface_default.xml',
    'eyes': HAAR_BASE_DIR + 'haarcascade_eye.xml',
    'smile': HAAR_BASE_DIR + 'haarcascade_smile.xml',
}

FACE_DETECTOR = dlib.get_frontal_face_detector()
FACE_PREDICTOR = dlib.shape_predictor(os.path.join(HAAR_BASE_DIR_USER, 'shape_predictor_68_face_landmarks.dat'))
FACE_PREDICTOR_SHAPE = (68, 2)


def download_classifiers():
    for name, url in HAAR_CASCADE_URLS.items():
        fpath = os.path.abspath(os.path.join(HAAR_BASE_DIR_USER, name + '.xml'))
        if os.path.isfile(fpath):
            continue
        log.info(f"Download haar cascade for {name} to {fpath}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fpath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)


# download_classifiers()
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_BASE_DIR, 'haarcascade_frontalface_default.xml'))
EYE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_BASE_DIR, 'haarcascade_eye.xml'))
EYE_GLASSES_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_BASE_DIR, 'haarcascade_eye_tree_eyeglasses.xml'))


def make_square(x, y, h, w):
    """
    Simple function that extends rectangle to a square
    """
    if h == w:
        return x, y, h, w
    diff = abs(h - w)
    if h > w:
        w = h
        y -= diff // 2
        if y < 0:
            y = 0
    else:
        h = w
        x -= diff // 2
        if x < 0:
            x = 0
    return x, y, h, w


def scale_rect(x, y, h, w, scale=1.0):
    if scale == 1:
        return x, y, h, w
    hs = int(h * scale)
    ws = int(w * scale)
    x -= (hs - h) // 2
    y -= (ws - w) // 2
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    return x, y, hs, ws


def __get_face(fpath):
    img = cv2.imread(fpath)
    gray = img
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for scale, neighbors in [(1, 0)]:  #[(1, 0), (2, 0), (4, 0), (1.2, 3), (1.3, 5), (1.4, 7), (2, 0), (4, 0)]:
        if neighbors == 0:
            # converting from dlib.rectangle to x,y,h,w
            faces = [(f.left(), f.top(), f.height(), f.width()) for f in FACE_DETECTOR(gray, scale)]
        else:
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors)
        if len(faces) == 1:
            return fpath, faces[0]
        if len(faces) == 0:
            continue
        faces_np = np.array(faces, dtype='uint16')
        face_area = faces_np[:, 2:4].sum(axis=1)
        largest_face = int(np.argmax(face_area))
        return fpath, faces[largest_face]
    return None


def detect_faces(images):
    pool = Pool()
    results = []
    detects = pool.imap(__get_face, images)
    for i, d in enumerate(detects):
        # for i, d in enumerate([__get_face(img) for img in images]):
        if d is not None:
            results.append(d)
        print(f'\rDetecting faces {i}/{len(images)}', end='')
    pool.close()
    detect_rate = len(results) / len(images) * 100
    print(f'\rDetected {len(results)} of {len(images)} ({detect_rate:.2f}%) faces')
    return results


def __get_face_features(args):
    fpath, rect = args
    img = cv2.imread(fpath)

    if rect is not None:
        x, y, h, w = rect
    else:
        x, y = 0, 0
        h = img.shape[1]
        w = img.shape[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
    gray = cv2.equalizeHist(gray)

    # rectangle args: left, top, right, bottom
    pred = FACE_PREDICTOR(gray, dlib.rectangle(x, y, x + w, y + h))
    coors = np.zeros(FACE_PREDICTOR_SHAPE, dtype="uint16")
    for i in range(pred.num_parts):
        coors[i] = (pred.part(i).x, pred.part(i).y)
    return fpath, coors


def extract_face_features(faces) -> np.ndarray:
    """
    Uses shape predictor to extract face features
    :param faces: list of tuple of ( img location, (x,y,h,w) of face rect )
    :return: matrix of coordinates
    """
    pool = Pool()
    results = np.zeros((len(faces),) + FACE_PREDICTOR_SHAPE, dtype="uint16")
    indices_map = {fname: i for i, (fname, rect) in enumerate(faces)}
    detects = pool.imap(__get_face_features, faces)
    for i, (fpath, coors) in enumerate(detects):
        results[indices_map[fpath],] = coors
        print(f'\rDetecting faces {i}/{len(faces)}', end='')
    pool.close()
    print(f'\r', end='')
    return results


def _get_eyes(fpath):
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_blurred = cv2.blur(gray, (3, 3))
    # detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=40)
    # results = []
    # if detected_circles is not None:
    #     detected_circles = np.uint16(np.around(detected_circles))
    #     for pt in detected_circles[0, :]:
    #         w, h = pt[2], pt[2]
    #         x = pt[0] - w // 2
    #         y = pt[1] - h // 2
    #         results.append((x, y, w, h))

    for scale, neighbors in [(1.8, 18), (1.6, 12), (1.4, 7), (1.3, 5), (1.2, 3)]:
        eyes = EYE_CASCADE.detectMultiScale(img, scaleFactor=scale, minNeighbors=neighbors)
        if len(eyes) >= 1:
            return fpath, eyes

    for scale, neighbors in [(1.8, 18), (1.6, 12), (1.4, 7), (1.3, 5), (1.2, 3)]:
            eyes = EYE_GLASSES_CASCADE.detectMultiScale(img, scaleFactor=scale, minNeighbors=neighbors)
            if len(eyes) >= 1:
                return fpath, eyes

    return fpath, []


def detect_eyes(images):
    pool = Pool()
    results = []
    detected = 0
    detects = pool.imap(_get_eyes, images)
    for i, d in enumerate(detects):
        if len(d[1]) > 0:
            detected += 1
            results.append(d)
        print(f'\rDetecting eyes {i}/{len(images)}', end='')
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
    log.info(f"Total detected eyes {detected} of {len(images)} ({detected / len(images) * 100:.2f}%)")
    return results
