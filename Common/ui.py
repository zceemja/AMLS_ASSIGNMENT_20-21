import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img_grid(mat, shape, width=1200, height=600):
    if isinstance(shape, tuple):
        shape_h = shape[0]
        shape_w = shape[1]
    else:
        shape_h = shape
        shape_w = shape

    W = width // shape_w
    H = height // shape_h

    for page in range(mat.shape[0]//(W*H)):
        grid = None
        for y in range(H):
            row = None
            for x in range(W):
                i = x + W * y + (page * W * H)
                row = mat[i, ] if row is None else np.hstack((row, mat[i, ]))
            grid = row if grid is None else np.vstack((grid, row))
        cv2.imshow('Face Detect', grid)
        while True:
            k = cv2.waitKey(33)
            if k > 0:
                break
        if k == 27:
            break
    cv2.destroyAllWindows()


def show_nn_history(history):
    epochs = range(1, len(history.epoch) + 1)
    if 'loss' in history.history:
        plt.plot(epochs, history.history['loss'], label='Training Loss')
    if 'accuracy' in history.history:
        plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    if 'val_accuracy' in history.history:
        plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()
