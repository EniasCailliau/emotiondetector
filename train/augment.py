import numpy as np
import scipy


def rotate_random(img):
    """Rotate an image between 0 and 15 degrees."""

    rotation = np.random.randint(0, 16)
    return scipy.ndimage.rotate(img, rotation, mode='nearest')




def flip_random(img):
    """Randomly flip an image up-down and/or left-right."""

    tmp = img.copy()

    if np.random.choice((0, 1)):
        tmp = np.fliplr(tmp)

    if np.random.choice((0, 1)):
        tmp = np.flipud(tmp)

    return tmp


def gamma_random(img):
    """Randomly set the gamma of an image between 0.5 and 2.0
    of its original value."""

    tmp = np.float32(img)
    inv_gamma = np.random.uniform(0.5, 2.0)
    tmp = 255 * (tmp / 255)**(1 / inv_gamma)
    tmp = np.uint8(tmp)
    return tmp





def augment_random(X,y):
    print("Feature shape after random augment {}".format(X.shape))
    print("Label shape after random augment {}".format(y.shape))
    feature_set= X.copy()
    labels= y.copy()
    for features,label in zip(X,y):
        transformations = [flip_random,gamma_random]
        np.random.shuffle(transformations)
        for tr in transformations:
            features=tr(features)
        feature_set = np.vstack((feature_set, features[None]))
        labels=np.concatenate(labels, label)
        break
    print("Feature shape after random augment {}".format(feature_set.shape))
    print("Label shape after random augment {}".format(labels.shape))
    return feature_set,labels


def augment_all(X,y):
    print("Feature shape after random augment {}".format(X.shape))
    print("Label shape after random augment {}".format(y.shape))
    feature_set= X.copy()
    labels= y.copy()
    for features,label in zip(X,y):
        transformations = [ flip_random,gamma_random]
        for tr in transformations:
            features = tr(features)
            feature_set = np.vstack((feature_set, features[None]))
            labels = np.append(labels, label)
    print("Feature shape after random augment {}".format(feature_set.shape))
    print("Label shape after random augment {}".format(labels.shape))
    return feature_set,labels
