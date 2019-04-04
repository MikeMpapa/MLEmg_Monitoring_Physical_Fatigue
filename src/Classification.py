import numpy
import sklearn
from scipy.spatial import distance
from hmmlearn import hmm
import keras
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Conv3D
from keras.models import Sequential

class kNN:
    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = k

    def classify(self, test_sample):
        n_classes = numpy.unique(self.Y).shape[0]
        y_dist = (distance.cdist(self.X,
                                 test_sample.reshape(1,
                                                     test_sample.shape[0]),
                                 'euclidean')).T
        i_sort = numpy.argsort(y_dist)
        P = numpy.zeros((n_classes,))
        for i in range(n_classes):
            P[i] = numpy.nonzero(self.Y[i_sort[0][0:self.k]] == i)[0].shape[0] / float(self.k)
        return (numpy.argmax(P), P)



def CNNclassifier(x_train, y_train,x_test, y_test,input_size,epochs=100,batch_size=64):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2),
                     activation='relu',
                     input_shape=(input_size,input_size,3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.BatchNormalization())
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0000001),
              metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              shuffle=False)
    return model


def trainHMM(features,lengths,components=1,covar='full'):
    return hmm.GaussianHMM(n_components=components, covariance_type=covar, n_iter=100).fit(features,lengths)

def trainGMM(features,components=1,covar='full'):

    [X,Y]=listOfFeatures2Matrix(features)
    gmm = sklearn.mixture.GaussianMixture(n_components=components, covariance_type=covar,max_iter=1000)
    gmm = gmm.fit(X)
    return gmm



def trainSVM(features, Cparam):
    '''
    Train a multi-class probabilitistic SVM classifier.
    Note:     This function is simply a wrapper to the sklearn functionality for SVM training
              See function trainSVM_feature() to use a wrapper on both the feature extraction and the SVM training (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - Cparam:           SVM parameter C (cost of constraints violation)
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:
        This function trains a linear-kernel SVM for a given C value. For a different kernel, other types of parameters should be provided.
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    svm = sklearn.svm.SVC(C = Cparam, kernel = 'linear',  probability = True)
    svm.fit(X,Y)

    return svm


def trainKNN(features, K):
    '''
    Train a kNN  classifier.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features.
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - K:                parameter K
    RETURNS:
        - kNN:              the trained kNN variable

    '''
    [Xt, Yt] = listOfFeatures2Matrix(features)
    knn = kNN(Xt, Yt, K)
    return knn


def trainRandomForest(features, n_estimators):
    '''
    Train a multi-class decision tree classifier.
    Note:     This function is simply a wrapper to the sklearn functionality for SVM training
              See function trainSVM_feature() to use a wrapper on both the feature extraction and the SVM training (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:
        This function trains a linear-kernel SVM for a given C value. For a different kernel, other types of parameters should be provided.
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators)
    rf.fit(X,Y)

    return rf


def trainGradientBoosting(features, n_estimators):
    '''
    Train a gradient boosting classifier
    Note:     This function is simply a wrapper to the sklearn functionality for SVM training
              See function trainSVM_feature() to use a wrapper on both the feature extraction and the SVM training (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:
        This function trains a linear-kernel SVM for a given C value. For a different kernel, other types of parameters should be provided.
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    rf = sklearn.ensemble.GradientBoostingClassifier(n_estimators = n_estimators)
    rf.fit(X,Y)

    return rf


def trainExtraTrees(features, n_estimators):
    '''
    Train a gradient boosting classifier
    Note:     This function is simply a wrapper to the sklearn functionality for extra tree classifiers
              See function trainSVM_feature() to use a wrapper on both the feature extraction and the SVM training (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:
        This function trains a linear-kernel SVM for a given C value. For a different kernel, other types of parameters should be provided.
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    et = sklearn.ensemble.ExtraTreesClassifier(n_estimators = n_estimators)
    et.fit(X,Y)

    return et



def trainSVM_RBF(features, Cparam):
    '''
    Train a multi-class probabilitistic SVM classifier.
    Note:     This function is simply a wrapper to the sklearn functionality for SVM training
              See function trainSVM_feature() to use a wrapper on both the feature extraction and the SVM training (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - Cparam:           SVM parameter C (cost of constraints violation)
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:
        This function trains a linear-kernel SVM for a given C value. For a different kernel, other types of parameters should be provided.
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    svm = sklearn.svm.SVC(C = Cparam, kernel = 'rbf',  probability = True)
    svm.fit(X,Y)

    return svm


def listOfFeatures2Matrix(features):
    '''
    listOfFeatures2Matrix(features)

    This function takes a list of feature matrices as argument and returns a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces
    '''

    X = numpy.array([])
    Y = numpy.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
    return (X, Y)


def normalizeFeatures(features):
    '''
    This function normalizes a feature set to 0-mean and 1-std.
    Used in most classifier trainning cases.

    ARGUMENTS:
        - features:    list of feature matrices (each one of them is a numpy matrix)
    RETURNS:
        - features_norm:    list of NORMALIZED feature matrices
        - MEAN:        mean vector
        - STD:        std vector
    '''
    X = numpy.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = numpy.vstack((X, f))
            count += 1

    MEAN = numpy.mean(X, axis=0) + 0.00000000000001;
    STD = numpy.std(X, axis=0) + 0.00000000000001;

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - MEAN) / STD
        features_norm.append(ft)
    return (features_norm, MEAN, STD)



def classifierWrapper(classifier, classifier_type, test_sample):
    '''
    This function is used as a wrapper to pattern classification.
    ARGUMENTS:
        - classifier:        a classifier object of type sklearn.svm.SVC or kNN (defined in this library) or sklearn.ensemble.RandomForestClassifier or sklearn.ensemble.GradientBoostingClassifier  or sklearn.ensemble.ExtraTreesClassifier
        - classifier_type:    "svm" or "knn" or "randomforests" or "gradientboosting" or "extratrees"
        - test_sample:        a feature vector (numpy array)
    RETURNS:
        - R:            class ID
        - P:            probability estimate

    EXAMPLE (for some audio signal stored in array x):
        import audioFeatureExtraction as aF
        import audioTrainTest as aT
        # load the classifier (here SVM, for kNN use load_model_knn instead):
        [classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step] = aT.load_model(model_name)
        # mid-term feature extraction:
        [mt_features, _, _] = aF.mtFeatureExtraction(x, Fs, mt_win * Fs, mt_step * Fs, round(Fs*st_win), round(Fs*st_step));
        # feature normalization:
        curFV = (mt_features[:, i] - MEAN) / STD;
        # classification
        [Result, P] = classifierWrapper(classifier, model_type, curFV)
    '''
    R = -1
    P = -1
    if classifier_type == "knn":
        [R, P] = classifier.classify(test_sample)
    elif classifier_type == "svm" or \
                    classifier_type == "randomforest" or \
                    classifier_type == "gradientboosting" or \
                    classifier_type == "extratrees" or \
                    classifier_type == "svm_rbf" or\
                    classifier_type == "gmm":
        R = classifier.predict(test_sample.reshape(1,-1))[0]
        P = classifier.predict_proba(test_sample.reshape(1,-1))[0]
    return [R, P]

