import sys
import glob
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import FeatureExtraction_1D as f1d
from pyAudioAnalysis import audioTrainTest as aT
import cPickle





def mtLabelExtraction(labels, Fs, mtWin, mtStep, stWin, stStep):
    mtWinRatio = int(round(mtWin / stStep))
    mtStepRatio = int(round(mtStep / stStep))

    mtLabels = []

    stLabels = stLabelsExtraction(labels, Fs, stWin, stStep)

    curPos = 0
    N = len(stLabels)
    while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curStLabels = stLabels[N1:N2]
            if curStLabels.count(0) > curStLabels.count(1):
                mtLabels.append(0)
            else:
                mtLabels.append(1)
            curPos += mtStepRatio

    return mtLabels, stLabels




def stLabelsExtraction(labels, Fs, Win, Step):

    Win = int(Win)
    Step = int(Step)

    N = len(labels)  # total number of samples
    curPos = 0
    countFrames = 0

    stLabels = []
    while (curPos + Win - 1 < N):  # for each short-term window until the end of signal
        countFrames += 1
        x = labels[curPos:curPos + Win]  # get current window

        if x.count(0) > x.count(1):
            stLabels.append(0)
        else:
            stLabels.append(1)

        curPos = curPos + Step  # update window position

    return stLabels



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


def featureExtraction(raw_data,time,gt_labels,mW,mS,sW,sS):
    #emg_features_vectors = []
    duration = float(time[-1]-time[0])
    Fs = round(len(raw_data) / duration)

    mtWin = mW
    mtStep = mS
    stWin = sW
    stStep = sS
    '''
    mtWin = 1
    mtStep = 0.25
    stWin = 0.13
    stStep = 0.04
    
    mtWin = 0.5
    mtStep = 0.25
    stWin = 0.2
    stStep = 0.1

    mtWin = 5.0
    mtStep = 1.0
    stWin = 0.5
    stStep = 0.5

    mtWin = 5.0
    mtStep = 1.0
    stWin = 0.13
    stStep = 0.04
    '''
    [MidTermFeatures, stFeatures] = f1d.mtFeatureExtraction(raw_data,Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))
    [MidTermLabels, stLabels] = mtLabelExtraction(gt_labels, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))


    return MidTermFeatures.copy(),MidTermLabels #, stFeatures,stLabels






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
        feature_vectors,gtWindowLabels = featureExtraction(emg_raw[-1], time[-1],gt_labels[-1],2,1,0.25,0.25)

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

    model =Classify(clf,featuresAll,bestParam)

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


def Classify(clf,featuresAll, bestParam):
    if clf == 'svm':
        model = aT.trainSVM(featuresAll, bestParam)
    elif clf == 'svm_rbf':
        model = aT.trainSVM_RBF(featuresAll, bestParam)
    elif clf == 'extratrees':
        model = aT.trainExtraTrees(featuresAll, bestParam)
    elif clf == 'randomforest':
        model = aT.trainRandomForest(featuresAll, bestParam)
    elif clf == 'knn':
        model = aT.trainKNN(featuresAll, bestParam)
    elif clf == 'gradientboosting':
        model = aT.trainGradientBoosting(featuresAll, bestParam)

    return model


if __name__ == '__main__':
    if sys.argv[1] == "-c":
        evaluateClassifier(sys.argv)

