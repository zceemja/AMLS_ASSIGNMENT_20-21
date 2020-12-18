import logging
import sys

import A1
import A2
import B1
import B2
import tensorflow as tf
import settings
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from Common.face_detect import setup_dependencies

if __name__ == '__main__':
    _log_root = logging.getLogger()
    _log_root.setLevel(logging.DEBUG)

    _log_ch = logging.StreamHandler(sys.stdout)
    _log_ch.setLevel(logging.DEBUG)
    _formatter = logging.Formatter('[%(asctime)s][%(name)s.%(funcName)s][%(levelname)s] %(message)s')
    _log_ch.setFormatter(_formatter)
    _log_root.addHandler(_log_ch)

    gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", gpus)
    if gpus == 0 and settings.THREADS > 0:
        tf.config.threading.set_intra_op_parallelism_threads(settings.THREADS)
    setup_dependencies()

    # ======================================================================================================================
    # Data preprocessing
    # data_train, data_val, data_test = common.data_preprocessing()
    # ======================================================================================================================
    # Task A1

    model_A1 = A1.Model(settings.DATASET_CELEBRA_IMG, settings.DATASET_CELEBRA_CSV)
    # if settings.TUNE_MODELS:
    #     model_A1.tune_model()
    # else:
    #     model_A1.model = SVC(C=1)
    # acc_A1_train = model_A1.train()
    acc_A1_train = model_A1.train_cnn()
    acc_A1_test = model_A1.test(settings.DATASET_CELEBRA_TEST_IMG, settings.DATASET_CELEBRA_TEST_CSV)
    model_A1.cleanup()
    # ======================================================================================================================
    # Task A2

    model_A2 = A2.Model(settings.DATASET_CELEBRA_IMG, settings.DATASET_CELEBRA_CSV)
    if settings.TUNE_MODELS:
        model_A2.tune_model()
    else:
        model_A2.model = SVC(C=10000, gamma=2.6826957952797274e-06)
    acc_A2_train = model_A2.train()
    acc_A2_test = model_A2.test(settings.DATASET_CELEBRA_TEST_IMG, settings.DATASET_CELEBRA_TEST_CSV)
    model_A2.cleanup()

    # ======================================================================================================================
    # Task B1
    model_B1 = B1.Model(settings.DATASET_CARTOON_IMG, settings.DATASET_CARTOON_CSV)
    if settings.TUNE_MODELS:
        model_B1.tune_model()
    else:
        model_B1.model = SVC(C=10, kernel='poly')
    acc_B1_train = model_B1.train()
    acc_B1_test = model_B1.test(settings.DATASET_CARTOON_TEST_IMG, settings.DATASET_CARTOON_TEST_CSV)
    model_B1.cleanup()

    # ======================================================================================================================
    # Task B2
    model_B2 = B2.Model(settings.DATASET_CARTOON_IMG, settings.DATASET_CARTOON_CSV)
    if settings.TUNE_MODELS:
        model_B2.tune_model()
    else:
        model_B2.model = RandomForestClassifier(n_estimators=300, n_jobs=settings.THREADS)
    acc_B2_train = model_B2.train()
    acc_B2_test = model_B2.test(settings.DATASET_CARTOON_TEST_IMG, settings.DATASET_CARTOON_TEST_CSV)
    model_B2.cleanup()

    # ======================================================================================================================
    ## Print out your results with following format:
    print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                            acc_A2_train, acc_A2_test,
                                                            acc_B1_train, acc_B1_test,
                                                            acc_B2_train, acc_B2_test))
