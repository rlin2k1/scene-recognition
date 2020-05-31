import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing, multiclass

from collections import defaultdict
import random
import sys
np.set_printoptions(threshold=sys.maxsize)

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    neigh = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(train_features, train_labels)
    predicted_categories = neigh.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    krnl = 'linear' if is_linear else 'rbf'
    g='scale'
    if krnl == 'rbf' and len(train_features):
        if len(train_features[0] == 128):
            if svm_lambda == .008:
                # print("g = .0004")
                g = .0004
            else:
                # print("g = .0003")
                g = .0003
        elif len(train_features[0] == 32):
            # print("g = .0011")
            g = .0011
    # According to the documentation, the default, ovr, "trains n_classes 
    # one-vs-rest classifiers", precisely fulfilling the spec's goal
    clf = multiclass.OneVsRestClassifier(svm.SVC(C=svm_lambda, kernel=krnl, gamma=g), n_jobs=-1)
    # print('Starting to fit')
    clf.fit(train_features, train_labels)
    # print('Starting to predict')
    predicted_categories = clf.predict(test_features)
    #print(predicted_categories[0:100])
    #print(len(predicted_categories))
    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    dim = (target_size, target_size)
    output_image = cv2.resize(input_image, dim)
    mean, std = cv2.meanStdDev(output_image)
    output_image -= mean[0]
    return output_image / std[0]

def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    return float(sum(1 for p, t in zip(true_labels, predicted_labels) if p == t)) / len(predicted_labels) * 100

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.
    desc = []
    if feature_type == "sift":
        for image in train_images:
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=25)
            _, des1 = sift.detectAndCompute(image,None)
            if des1 is None:
                continue
            for i in des1:
                desc.append(i)
    elif feature_type == "surf":
        for image in train_images:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=500, extended=True)
            _, des1 = surf.detectAndCompute(image,None)
            if des1 is None:
                continue
            nf = 25
            if len(des1) > nf:
                random.shuffle(des1)
                des1 = des1[:nf]
            for i in des1:
                desc.append(i)
    elif feature_type == "orb":
        for image in train_images:
            orb = cv2.ORB_create(nfeatures=25)
            _, des1 = orb.compute(image, orb.detect(image, None))
            if des1 is None:
                continue
            for i in des1:
                desc.append(i)
    else:
        return None

    vocabulary = [[]] * dict_size
    if clustering_type == "kmeans":
        kmeans = cluster.KMeans(n_clusters=dict_size, n_jobs=4).fit(desc) #, n_jobs=-1).fit(desc)
        vocabulary = kmeans.cluster_centers_
    elif clustering_type == "hierarchical":
        aggc = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(desc)
        lmap = defaultdict(list)
        for idx, l in enumerate(aggc.labels_):
            lmap[l].append(desc[idx])
        cluster_avgs = [np.mean(lmap[n], axis=0) for n in range(dict_size)]
        distances = [float('inf')] * dict_size
        for idx, l in enumerate(aggc.labels_):
            dist = np.linalg.norm(desc[idx] - cluster_avgs[l])
            if dist < distances[l]:
                distances[l] = dist
                vocabulary[l] = desc[idx]
    else:
        return None

    return vocabulary

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    des1 = []
    bow = [0.0] * len(vocabulary)
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
        _, des1 = sift.detectAndCompute(image,None)
        if des1 is None:
            return bow
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create(extended=True)
        _, des1 = surf.detectAndCompute(image,None)
        if des1 is None:
            return bow
    elif feature_type == "orb":
        orb = cv2.ORB_create()
        _, des1 = orb.compute(image, orb.detect(image, None))
        if des1 is None:
            return bow
    else:
        return bow

    for x in des1:
        bow[np.array(np.linalg.norm(x - vocabulary, axis=1)).argmin()] += 1
    # compared this normalization strategy to dividing by the total number of words
    # in bow representation and this performed better
    #npbow = np.array(bow).astype(np.float32)
    #return npbow / np.sum(bow)
    return np.array(bow) * 65536.0 / (len(image) * len(image[0]))

def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    sizes = [8, 16, 32]
    neighbors = [1, 3, 6]

    classResult = []

    for size in sizes:
        for neighbor in neighbors:
            start = timeit.default_timer()
            trainf = [imresize(t.astype(np.float32)/255, size).flatten() for t in train_features]
            testf = [imresize(t.astype(np.float32)/255, size).flatten() for t in test_features]
            predicted = KNN_classifier(trainf, train_labels, testf, neighbor)
            classResult.append(reportAccuracy(test_labels, predicted))
            classResult.append(timeit.default_timer() - start)
    return classResult