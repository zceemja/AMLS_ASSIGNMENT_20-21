from os import path

# Stop and show
SHOW_GRAPHS = True

TUNE_MODELS = False
# Number of threads used in multithreaded functions. -1 for all.
THREADS = -1

# Directory for generated/cached data or model/classifier
DATA_PATH = 'Data'

# Directory where haar cascades are located. If not in there, they will be downloaded in DATA_PATH
HAAR_DIR = "/usr/share/opencv4/haarcascades"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"

# Dlib facial landmark predictor file location
DLIB_PREDICTOR = path.join(DATA_PATH, "shape_predictor_68_face_landmarks.dat")
DLIB_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"


DATASET_CARTOON = 'Datasets/cartoon_set'
DATASET_CARTOON_IMG = path.join(DATASET_CARTOON, 'img')
DATASET_CARTOON_CSV = path.join(DATASET_CARTOON, 'labels.csv')

DATASET_CARTOON_TEST = 'Datasets/cartoon_set_test'
DATASET_CARTOON_TEST_IMG = path.join(DATASET_CARTOON_TEST, 'img')
DATASET_CARTOON_TEST_CSV = path.join(DATASET_CARTOON_TEST, 'labels.csv')

DATASET_CELEBRA = 'Datasets/celeba'
DATASET_CELEBRA_IMG = path.join(DATASET_CELEBRA, 'img')
DATASET_CELEBRA_CSV = path.join(DATASET_CELEBRA, 'labels.csv')

DATASET_CELEBRA_TEST = 'Datasets/celeba_test'
DATASET_CELEBRA_TEST_IMG = path.join(DATASET_CELEBRA_TEST, 'img')
DATASET_CELEBRA_TEST_CSV = path.join(DATASET_CELEBRA_TEST, 'labels.csv')
