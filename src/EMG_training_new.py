import sys
import glob
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import FeatureExtraction_1D as f1d
from pyAudioAnalysis import audioTrainTest as aT
import cPickle



def showEMGData(data,duration,gt):

    Fs = round(len(data) / float(duration))
    fatigue_thresh = gt.index(1)

    T_no_fatigue = np.arange(0, len(data[:fatigue_thresh]) / float(Fs), 1.0 / Fs)
    T_fatigue = np.arange(0, len(data) / float(Fs), 1.0 / Fs)

    if len(T_no_fatigue) > len(data[:fatigue_thresh]):
        T_no_fatigue = T_no_fatigue[:-1]
    if len(T_fatigue) > len(data):
            T_fatigue = T_fatigue[:-1]
    plt.plot(T_fatigue, data,'--r')
    plt.plot(T_no_fatigue, data[:fatigue_thresh])
    plt.show()


def featureExtraction(raw_data,time,gt_labels):
    #emg_features_vectors = []
    duration = float(time[-1]-time[0])
    Fs = round(len(raw_data) / duration)

    mtWin = 0.5
    mtStep = 0.25
    stWin = 0.2
    stStep = 0.1
    '''
    mtWin = 5.0
    mtStep = 1.0
    stWin = 0.5
    stStep = 0.5
    '''
    [MidTermFeatures, stFeatures] = f1d.mtFeatureExtraction(raw_data, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))

    emg_features_vectors = MidTermFeatures.copy()

    # Assign labels to mid-term windows

    #numOfmtWindows = int(round(len(gt_labels) / round(mtStep * Fs)))
    numOfmtWindows =  MidTermFeatures.shape[1]
    '''
    print numOfmtWindows,MidTermFeatures.shape,
    if MidTermFeatures.shape[1]>numOfmtWindows:
        numOfmtWindows +=1
    print numOfmtWindows,MidTermFeatures.shape
    '''
    gt_WindowLabels = []
    cur = 0
    N = int( round(mtStep * Fs))
    for w in range(numOfmtWindows):
            if N > len(gt_labels):
                N = len(gt_labels) - 1
            c0 = gt_labels[cur:N].count(0)  # count no Fatigue labels
            c1 = gt_labels[cur:N].count(1)  # count Fatigue labels
            #print w,c0,c1, gt_labels[cur:N]
            if c0 > c1:
                gt_WindowLabels.append(0)
            else:
                gt_WindowLabels.append(1)
            cur = N
            N = N + int( round(mtStep * Fs))

    return emg_features_vectors,gt_WindowLabels






def evaluateClassifier(argv):
    save = argv[5]

    dirName = argv[2]    # path to csv files
    fileList  = sorted(glob.glob(os.path.join(dirName, "*.csv")))

    #data = {}
    #data['user'] = {}
    user = []
    exercise = []
    repetition = []
    time = []
    emg_raw = []
    gt_labels = []
    feature_vectors_nofatigue = []
    feature_vectors_fatigue = []

    for file in fileList:

        with open(file,'r') as f:
            x = f.readlines()
            if not x:
                continue
            time.append( [float(label.split(',')[0]) for label in x ])
            emg_raw.append( [float(label.split(',')[1]) for label in x ])
            gt_labels.append( [int(label.split(',')[2].rstrip()) for label in x ])
        f.close

        #split the sample into the positive and negative classes
        ###
        feature_vectors,gtWindowLabels = featureExtraction(emg_raw[-1], time[-1],gt_labels[-1])

        for i,w in enumerate(gtWindowLabels):
            if w==0:
                feature_vectors_nofatigue.append( feature_vectors[:,i])
            else:
                feature_vectors_fatigue.append(feature_vectors[:,i])


        user.append(file.split('/')[-1].split('E')[0][1:])
        exercise.append(file.split('/')[-1].split('R')[0][-1])
        repetition.append(file.split('/')[-1].split('.')[0][-1])


        if argv[-1] == '-s':
            showEMGData(emg_raw[-1],time[-1][-1]-time[-1][0],gt_labels[-1])


    #Collect all features
    featuresAll = []
    featuresAll.append(np.array(feature_vectors_nofatigue))
    featuresAll.append(np.array(feature_vectors_fatigue))
    labelsAll = ['0:NoFtigue','1:Fatigue'] # 0:NoFtigue, 1:Fatigue

    #Normilize features
    (featuresAll, MEAN, STD) = aT.normalizeFeatures(featuresAll)

    clf = argv[3][1:]
    params = argv[4]
    bestParam = aT.evaluateclassifier(featuresAll, labelsAll, 1000, clf , params, 0, perTrain=0.80)


    MEAN = MEAN.tolist()
    STD = STD.tolist()

    if clf == 'svm':
        model = aT.trainSVM(featuresAll, bestParam)
    elif   clf == 'svm_rbf':
        model = aT.trainSVM_RBF(featuresAll, bestParam)
    elif clf == 'extratrees':
        model = aT.trainExtraTrees(featuresAll, bestParam)
    elif clf == 'randomforest':
        model = aT.trainRandomForest(featuresAll, bestParam)
    elif clf == 'knn':
        model = aT.trainKNN(featuresAll, bestParam)
    elif clf == 'gradientboosting':
        model = aT.trainGradientBoosting(featuresAll, bestParam)


    if save:
        saveClassifier(clf,bestParam,model,MEAN,STD,labelsAll)

    print 'Training of',clf,'completed'

    return clf,model,labelsAll,MEAN,STD,bestParam



def saveClassifier(clf_name,bestParam,model,MEAN,STD,labelsAll):
    # STEP C: Save the classifier to file
    modelName = clf_name + '_' + str(bestParam)
    with open(modelName, 'wb') as fid:  # save to file
        cPickle.dump(model, fid)
    fo = open(modelName + "MEANS", "wb")
    cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(labelsAll, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()



if __name__ == '__main__':
    if sys.argv[1] == "-c":
        evaluateClassifier(sys.argv)

