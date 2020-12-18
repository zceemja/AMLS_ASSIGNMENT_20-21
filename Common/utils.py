from os import path
import pickle

from sklearn.model_selection import cross_val_score
import numpy as np
import settings


def cache(file_name, load_func, *func_args, **func_kwargs):
    """
    Caches result from any function
    :param file_name: name for caching
    :param load_func: function that outputs result (if cache not available)
    :return: result from function
    """
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        data = load_func(*func_args, **func_kwargs)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
        return data


def find_best_model(args, train, test, cv=5):
    highest_score = 0
    highest_score_model = None
    for arg in args:
        model = arg['model']
        del arg['model']
        model = model(**arg)
        scores = cross_val_score(model, train, test, cv=cv, n_jobs=None if 'n_jobs' in arg else settings.THREADS)
        # elif isinstance(model, list):
        #     kf = KFold(n_splits=cv)
        #     scores = np.zeros((cv, ))
        #     for i, (train_index, test_index) in enumerate(kf.split(train)):
        #         X_train, X_test = train[train_index], train[test_index]
        #         y_train, y_test = test[train_index], test[test_index]
        #
        #         # Build same model each time.
        #         # Could setup stateful model to be able to reset same model without rebuilding.
        #         kmodel = keras.Sequential(model)
        #         kmodel.compile(**arg.get('compile', {}))
        #         kmodel.fit(X_train, y_train, validation_data=(X_test, y_test), **arg.get('fit', {}))
        #         scores[i] = kmodel.evaluate(X_test, y_test)[1]
        #         kmodel = None
        score = scores.mean()
        print(f"Model {model} scored {score:.5f} (+/- {scores.std()*2:.5f})")
        if score > highest_score:
            highest_score = score
            highest_score_model = model
    print(f"Highest accuracy model: {highest_score_model}")
    return highest_score_model


def binary_to_one_hot(arr: np.ndarray) -> np.ndarray:
    """
    Converts binary (l, 1) array to one-hot (l, 2) array
    """
    res = np.zeros((arr.shape[0], 2))
    res[np.where(arr == 1)[0], 0] = 1
    res[np.where(arr == 0)[0], 1] = 1
    return res


def max_arg_one_hot(arr: np.ndarray) -> np.ndarray:
    """
    Converts likelihood array back to one hot depending on highest value
    """
    # using argmin because inverted indices in binary_to_one_hot
    return binary_to_one_hot(np.argmin(arr, axis=1))

