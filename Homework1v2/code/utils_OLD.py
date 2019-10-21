import cv2
import numpy as np
import timeit
import os
import sys
from collections import defaultdict
from sklearn import neighbors, svm, cluster, multiclass
import pickle
import random
np.set_printoptions(threshold=sys.maxsize)

def imresize(input_image, target_size):
    # resizes the input image to a new image of size [target_size, target_size]. normalizes the output image
    # to be zero-mean with unit variance
    dim = (target_size, target_size)
    output_image = cv2.resize(input_image, dim)
    mean, std = cv2.meanStdDev(output_image)
    output_image -= mean[0]
    return output_image / std[0]

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
    return float(sum(1 for p, t in zip(true_labels, predicted_labels) if p == t)) / len(predicted_labels)

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a n x 1 array of images
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be dict_size x d, where d is the 
    # dimension of the feature. each row is a cluster centroid / visual word.
    
    # Alex noted during discussion that only kmeans returns centroids and that for AHC, we must
    # determine the centroid by finding the nearest neighbor to the cluster avg
    # Write and read from file to save time.
    desc = []
    if feature_type == "sift":
        for image in train_images:
            sift = cv2.xfeatures2d.SIFT_create() if clustering_type == "kmeans" else cv2.xfeatures2d.SIFT_create(nfeatures=27)
            _, des1 = sift.detectAndCompute(image,None)
            if des1 is None:
                continue
            for i in des1:
                desc.append(i)
    elif feature_type == "surf":
        for image in train_images:
            surf = cv2.xfeatures2d.SURF_create(extended=True) if clustering_type == "kmeans" else cv2.xfeatures2d.SURF_create(hessianThreshold=500, extended=True)
            _, des1 = surf.detectAndCompute(image,None)
            if des1 is None:
                continue
            nf = 27
            if len(des1) > nf and clustering_type != "kmeans":
                random.shuffle(des1)
                des1 = des1[:nf]
            for i in des1:
                desc.append(i)
    elif feature_type == "orb":
        for image in train_images:
            orb = cv2.ORB_create() if clustering_type == "kmeans" else cv2.ORB_create(nfeatures=30)
            _, des1 = orb.compute(image, orb.detect(image, None))
            if des1 is None:
                continue
            for i in des1:
                desc.append(i)
    else:
        return None

    print(len(desc))
    vocabulary = [[]] * dict_size
    if clustering_type == "kmeans":
        kmeans = cluster.KMeans(n_clusters=dict_size).fit(desc) #, n_jobs=-1).fit(desc)
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

    print(len(vocabulary))
    print(len(vocabulary[0]))
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

def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an n x d matrix, where d is the dimensionality of
    # the feature representation.
    # train_labels is an n x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an m x d matrix, where d is the dimensionality of the
    # feature representation. (you can assume m=n unless you modified the 
    # starter code)
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # lambda is a scalar, the value of the regularizer for the SVMs
    # predicted_categories is an m x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    krnl = 'linear' if is_linear else 'rbf'
    g='scale'
    if krnl == 'rbf' and len(train_features):
        if len(train_features[0] == 128):
            if svm_lambda == .008:
                print("g = .0004")
                g = .0004
            else:
                print("g = .0003")
                g = .0003
        elif len(train_features[0] == 32):
            print("g = .0011")
            g = .0011
    # According to the documentation, the default, ovr, "trains n_classes 
    # one-vs-rest classifiers", precisely fulfilling the spec's goal
    clf = multiclass.OneVsRestClassifier(svm.SVC(C=svm_lambda, kernel=krnl, gamma=g), n_jobs=-1)
    print('Starting to fit')
    clf.fit(train_features, train_labels)
    print('Starting to predict')
    predicted_categories = clf.predict(test_features)
    #print(predicted_categories[0:100])
    #print(len(predicted_categories))
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
            trainf = [imresize(t.astype(np.float32)/255, size).flatten() for t in train_features]
            testf = [imresize(t.astype(np.float32)/255, size).flatten() for t in test_features]
            predicted = KNN_classifier(trainf, train_labels, testf, neighbor)
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
                    train_features.append(cv2.imread(os.path.join(folder, f, file), cv2.IMREAD_GRAYSCALE))
                    train_labels.append(index)
                elif tt == 'test':
                    test_features.append(cv2.imread(os.path.join(folder, f, file), cv2.IMREAD_GRAYSCALE))
                    test_labels.append(index)
    tinyRes = tinyImages(train_features, test_features, train_labels, test_labels)
    print(tinyRes)
    sys.exit(1)

    print('Done reading in all images')

    fname = "../orb30h50.pkl"
    # If there's a saved vocabulary, assume everything is good and use it for classification
    if os.path.exists(fname):
        vocab = []
        with open(fname, 'rb') as f:
            vocab = pickle.load(f)
        print("Reusing saved buildDict output")
        trfsname = "../trfs_" + fname[3:]
        tefsname = "../tefs_" + fname[3:]
        train_fs = []
        test_fs = []
        if os.path.exists(trfsname) and os.path.exists(tefsname):
            with open(trfsname, 'rb') as f:
                train_fs = pickle.load(f)
            with open(tefsname, 'rb') as f:
                test_fs = pickle.load(f)
            print("Reusing saved BOW representations")
        else:
            train_fs = [computeBow(tf, vocab, 'orb') for tf in train_features]
            test_fs = [computeBow(tf, vocab, 'orb') for tf in test_features]
            pickle.dump(train_fs, open(trfsname, 'wb'))
            pickle.dump(test_fs, open(tefsname, 'wb'))
        accuracy = []
        runtime = []
        lin = False
        for c in [1.0 + .5 * x for x in range(1, 12)]: #[.0001 * (10**x) for x in range(6)]: #[2.7 + .1 * x for x in range(8)]: #[1.06 + .1 * x for x in range(20)]:
            #for g in [.0001 * x for x in range(1, 10)]: #[.000001 * (10**x) for x in range(8)]: #[38 + 10 * x for x in range(5)]: #['scale']: #[1.0 * x for x in range(1,75)]: #[.00001 * (10**x) for x in range(7)]: #['scale']: #[.0001 * x for x in range(1, 8)]:
            #print("C = {}, gamma = {}".format(c, g))
            print("C = {}".format(c))
            start = timeit.default_timer()
            predicted = SVM_classifier(train_fs, train_labels, test_fs, lin, c)
            #predicted = KNN_classifier(train_fs, train_labels, test_fs, 6)
            acc = reportAccuracy(test_labels, predicted)
            print(acc)
            accuracy.append(acc)
            runtime.append(timeit.default_timer() - start)
        print(accuracy)
        print(runtime)
        print(sorted(accuracy))
        print(max(accuracy))
        sys.exit(1)
    
    #print(tinyImages(train_features, test_features, train_labels, test_labels, label_dict))
    dict_size = 50
    feature_type = "surf"
    clustering_type = "kmeans"
    vocab = buildDict(train_features, dict_size, feature_type, clustering_type)
    pickle.dump(vocab, open("../surfextinfk50.pkl", "wb" ))
    

if __name__ == "__main__":
    main()