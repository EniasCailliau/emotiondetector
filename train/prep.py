import numpy as np
from keras.utils import np_utils
from skimage.io import imread
from skimage.measure import block_reduce

from utils import list_all_files
from utils.general import dump_pickle, load_pickle
from train import augment

DEFAULT_DATASET_PATH = "data/dataset.pkl"

def faces_dataset(examples, block_size=1):
    X = []
    y = []
    for path, label in examples:
        img = imread(path, as_grey=True)
        img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
        X.append(img)
        y.append(label)
    return np.asarray(X), np.asarray(y)


def load_faces_dataset():
    negative_paths = list(list_all_files('../faces/dbase/negatives/negatives7/', ['.jpg']))
    print('loaded', len(negative_paths), 'negative examples')
    positive_paths = list(list_all_files('../faces/dbase/positives/positives7/', ['.jpg']))
    print('loaded', len(positive_paths), 'positive examples')
    examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]
    X, y = faces_dataset(examples)

    X = X.astype(np.float32) / 255.
    y = y.astype(np.int32)
    print(X.dtype, X.min(), X.max(), X.shape)
    print(y.dtype, y.min(), y.max(), y.shape)

    from utils import make_mosaic, show_array
    show_array(255 * make_mosaic(X[:len(negative_paths)], 8), fmt='jpeg')  # negative at the beginning
    show_array(255 * make_mosaic(X[-len(positive_paths):], 8), fmt='jpeg')  # positive at the end

    X = np.expand_dims(X, axis=-1)
    return X, y

def pickle_faces_dataset():
    X,y=load_faces_dataset()
    X_all, y_all =augment.augment_all(X,y)

    dump_pickle(dict(X=X_all,y=y_all), path=DEFAULT_DATASET_PATH)

def unpickle_faces_dataset():
    dataset=load_pickle(DEFAULT_DATASET_PATH)
    return dataset['X'],dataset['y']

def prepare_data(X, y):
    # convert classes to vector
    nb_classes = 2
    y_orig = y
    y = np_utils.to_categorical(y, nb_classes).astype(np.float32)

    # shuffle all the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # prepare weighting for classes since they're unbalanced
    class_totals = y.sum(axis=0)
    class_weight = class_totals.max() / class_totals

    print(X.dtype, X.min(), X.max(), X.shape)
    print(y.dtype, y.min(), y.max(), y.shape)
    return X, y, y_orig, class_weight

if __name__ == "__main__":
    pickle_faces_dataset()