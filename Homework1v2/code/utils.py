import cv2
import numpy as np
import timeit
import os
import sys
from sklearn import neighbors, svm, cluster
np.set_printoptions(threshold=sys.maxsize)

def imresize(input_image, target_size):
    # resizes the input image to a new image of size [target_size, target_size]. normalizes the output image
    # to be zero-mean with unit variance
    dim = (target_size, target_size)
    output_image = cv2.resize(input_image, dim)
    # this version ensures that the range is between -1 and 1 but does not ensure the output is 0 mean
    #output_image = cv2.normalize(output_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mean, std = cv2.meanStdDev(output_image)
    output_image -= mean[0]
    output_image /= std[0]
    return output_image

def reportAccuracy(true_labels, predicted_labels, label_dict = None):
    # generates and returns the accuracy of a model
    # true_labels is a n x 1 cell array, where each entry is an integer
    # and n is the size of the testing set.
    # predicted_labels is a n x 1 cell array, where each entry is an 
    # integer, and n is the size of the testing set. these labels 
    # were produced by your system
    # label_dict is a 15x1 cell array where each entry is a string
    # containing the name of that category. WE DO NOT NEED THIS. ONLY FOR TA USE
    # accuracy is a scalar, defined in the spec (in %)
    num_correct_predictions = 0
    num_predictions = len(predicted_labels)
    for i in range(num_predictions):
        if true_labels[i] == predicted_labels[i]:
            num_correct_predictions = num_correct_predictions + 1
    # doing this conversion so it works with Python 2 as well
    accuracy = float(num_correct_predictions) / num_predictions
    return accuracy

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a n x 1 array of images
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be dict_size x d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.
    return vocabulary

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary
    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    return Bow

def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation. You can assume M = N unless you've modified the
    # starter code.
    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    neigh = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(train_features, train_labels)
    predicted_categories = neigh.predict(test_features)
    return predicted_categories

def tinyImages(train_features, test_features, train_labels, test_labels, label_dict = None):
    # train_features is a nx1 array of images
    # test_features is a nx1 array of images

    # train_labels is a nx1 array of integers, containing the label values
    # test_labels is a nx1 array of integers, containing the label values

    # label_dict is a 15x1 array of strings, containing the names of the labels
    # classResult is a 18x1 array, containing accuracies and runtimes
    # We have NINE Tests. With 2 Values: Accuracy and Runtime for a Total of 18.
    sizes = [8, 16, 32]
    neighbors = [1, 3, 6]

    accuracy = []
    runtime = []

    for size in sizes:
        for neighbor in neighbors:
            start = timeit.default_timer()
            train_resize = []
            for train in train_features:
                resize = np.amin(imresize(train, size), axis=2).flatten()
                train_resize.append(resize)
            test_resize = []
            for test in test_features:
                resize = np.amin(imresize(test, size), axis=2).flatten()
                test_resize.append(resize)
            predicted = KNN_classifier(train_resize, train_labels, test_resize, neighbor)
            accuracy.append(reportAccuracy(test_labels, predicted))
            runtime.append(timeit.default_timer() - start)
    
    classResult = accuracy + runtime
    return classResult
    
# ---------------------------------------------------------------------------- #
# TESTING
# ---------------------------------------------------------------------------- #
def main():
    # image = cv2.imread('../data/train/bedroom/image_0001.jpg').astype(np.float32) / 255
    # cv2.imshow('Unchanged', image)
    # cv2.waitKey(0)
    # cv2.imshow('Changed', imresize(image, 8))
    # cv2.waitKey(0)
    # cv2.imshow('Changed', imresize(image, 16))
    # cv2.waitKey(0)
    # cv2.imshow('Changed', imresize(image, 32))
    # cv2.waitKey(0)
    #-----------------------------------------------------------------------------------------------

    rootdir = os.getcwd()[:-4] + 'data'
    train_features = []
    test_features = []
    train_labels = []
    test_labels = []
    # Slice Label Dict to Improve Testing Speed
    label_dict = sorted([x.lower() for x in os.listdir(rootdir + '/train')])
    for tt in os.listdir(rootdir):
        folder = os.path.join(rootdir, tt)
        for f in os.listdir(folder):
            if f.lower() not in label_dict:
                continue
            for file in os.listdir(os.path.join(folder, f)):
                index = label_dict.index(f.lower())
                if tt == 'train':
                    train_features.append(cv2.imread(os.path.join(folder, f, file)).astype(np.float32) / 255)
                    train_labels.append(index)
                elif tt == 'test':
                    test_features.append(cv2.imread(os.path.join(folder, f, file)).astype(np.float32) / 255)
                    test_labels.append(index)
                    
    # print(label_dict)
    # print(len(label_dict))
    # print(test_labels)
    # print(train_labels)

    print(tinyImages(train_features, test_features, train_labels, test_labels, label_dict))

if __name__ == "__main__":
    main()
