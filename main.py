import settings
from Common import common
import A1
import A2
import B1
import B2
from os.path import join
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# ======================================================================================================================
# Data preprocessing
data_train, data_val, data_test = common.data_preprocessing()
# ======================================================================================================================
# Task A1
# Build model object.
model_A1 = A1.Model(settings.DATASET_CELEBRA_IMG, settings.DATASET_CELEBRA_CSV)
model_A1.tune_model()
acc_A1_train = model_A1.train()
acc_A1_test = model_A1.test()
model_A1.cleanup()
exit()
# ======================================================================================================================
# Task A2
# model_A2 = A2.Model(join(DATASET_CELEBRA, 'img'), join(DATASET_CELEBRA, 'labels.csv'))
# acc_A2_train = model_A2.train()
# acc_A2_test = model_A2.test()
# model_A2.cleanup()

# ======================================================================================================================
# Task B1
model_B1 = B1.Model(settings.DATASET_CARTOON_IMG, settings.DATASET_CARTOON_CSV)
model_B1.tune_model()
acc_B1_train = model_B1.train()
acc_B1_test = model_B1.test()
model_B1.cleanup()
#
# # ======================================================================================================================
# # Task B2
model_B2 = B2.Model(settings.DATASET_CARTOON_IMG, settings.DATASET_CARTOON_CSV)
model_B2.tune_model()
acc_B2_train = model_B2.train()
acc_B2_test = model_B2.test()
model_B2.cleanup()

# ======================================================================================================================
## Print out your results with following format:
# print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                         acc_A2_train, acc_A2_test,
#                                                         acc_B1_train, acc_B1_test,
#                                                         acc_B2_train, acc_B2_test))
