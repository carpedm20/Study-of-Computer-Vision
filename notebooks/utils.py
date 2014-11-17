import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import zoom

from skimage import color, feature, filter
from skimage.io import imread
from skimage.transform import resize

from sklearn.cross_validation import train_test_split

import os
import time
import glob
import pyprind
from random import shuffle
import multiprocessing as mp

orientations = 9
cell = 16
block = 3

SIZE = (30,15)
TEST = True

if TEST:
    COUNT = 20
else:
    COUNT = len(foods)

print "[*] Test mode : %s" % TEST

fig = plt.figure(figsize=SIZE)

FOOD_PATH = "/home/carpedm20/data/food100/"

if TEST:
    food1 = glob.glob("/home/carpedm20/data/food100/1/*.jpg")
    print "\nfood1 : %s" % len(food1)
    food2 = glob.glob("/home/carpedm20/data/food100/36/*.jpg")
    print "food2 : %s" % len(food2)
    food3 = glob.glob("/home/carpedm20/data/food100/23/*.jpg")
    print "food3 : %s" % len(food3)
    foods = food1 + food2 + food3
else:
    foods = glob.glob("/home/carpedm20/data/food100/*/*.jpg")

def build_labels(foods):
    new_foods = []
    for food in foods:
        food_label = food[len(FOOD_PATH):].split("/")[0]
        new_foods.append((food, food_label))
    return new_foods

foods = build_labels(foods)

train, test = train_test_split(foods, test_size=0.33, random_state=42)

shuffle(train)
shuffle(test)

print "\ntrain : %s" % len(train)
print "test : %s" % len(test)

train_labels = [int(x[1]) for x in train]
test_labels = [int(x[1]) for x in test]

def get_foods(foods, limit=COUNT):
    bar = pyprind.ProgBar(limit, monitor=True)
    for idx, food in enumerate(foods[:limit]):
        bar.update()
        img = imread(food[0])

        try:
            yield color.rgb2gray(img)
        except:
            continue

def get_cropped_foods(foods, height, width, limit=COUNT):
    bar = pyprind.ProgBar(limit, monitor=True)
    for idx, food in enumerate(foods[:limit]):
        bar.update()
        img = imread(food[0])
        img = resize_image(img, (height, width))

        try:
            yield color.rgb2gray(img)
        except:
            continue

def build_feature_vecture(foods):
    fs = []
    #hogs = []
    for food in get_cropped_foods(foods, 256, 256):
        f = feature.hog(food,
                        orientations=orientations,
                        pixels_per_cell=(cell, cell),
                        cells_per_block=(block, block))
                        #visualise=True)
        #dimension = len(f)
        fs.append((f.ravel()).tolist())
        #hogs.append(hog)
    return fs
        
def oversample(images, crop_dims):
    """
    code from https://github.com/BVLC/caffe/blob/master/python/caffe/io.py

    Crop images into the four corners, center, and their mirrored versions.

    Take
    image: iterable of (H x W x K) ndarrays
    crop_dims: (height, width) tuple for the crops.

    Give
    crops: (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
                            im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops


def resize_image(im, new_dims, interp_order=1):
    """
    code from https://github.com/BVLC/caffe/blob/master/python/caffe/io.py

    Resize an image array with interpolation.

    Take
    im: (H x W x K) ndarray
    new_dims: (height, width) tuple of new dimensions.
    interp_order: interpolation order, default is linear.

    Give
    im: resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        # skimage is fast but only understands {1,3} channel images in [0, 1].
        im_min, im_max = im.min(), im.max()
        im_std = (im - im_min) / (im_max - im_min)
        resized_std = resize(im_std, new_dims, order=interp_order)
        resized_im = resized_std * (im_max - im_min) + im_min
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def print_file_info(fname):
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(fname)
    print "\n============================="
    print "last modified: %s" % time.ctime(mtime)
    print "size: %s" % sizeof_fmt(size)
    print "=============================\n"

import cPickle

def save_pickle(variable, fname):
    make = True

    if os.path.isfile(fname):
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(fname)

        print "There exists %s" % fname
        print_file_info(fname)
        print "Will you overwrite it? [Y/n] : ",
        y_or_n = raw_input()

        if y_or_n == "Y" or y_or_n == "y" or y_or_n == "\n":
            make = True
        else:
            make = False

    if make:
        with open(fname, 'wb') as f:
            cPickle.dump(variable, f) 
        print_file_info(fname)

def load_pickle(fname):
    if os.path.isfile(fname):
        print_file_info(fname)
        with open(fname, 'rb') as fid:
            fs = cPickle.load(fid)
    else:
        print "There is no %s" % fname

def get_image_by_class(foods, idx):
    classes = list(set([food[1] for food in foods]))

    tmp = [food for food in foods if food[1] == classes[idx]]
    shuffle(tmp)

    return tmp[0][0]

def classify(clf, fname):
    img = imread(fname)
    answer = fname.split('/')[-2]

    start_time = time.time()

    img = resize_image(img, (256, 256))
    img = color.rgb2gray(img)
    
    f = feature.hog(img,
                    orientations=orientations,
                    pixels_per_cell=(cell, cell),
                    cells_per_block=(block, block))

    print "result : %s -> %s" % (answer, clf.predict(f)[0])

    end_time = time.time()
    print ("%g seconds" % (end_time - start_time))

