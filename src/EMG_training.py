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


def featureExtraction(raw_data,time):
    #emg_features_vectors = []
    for idx,sample in enumerate(raw_data):
        duration = float(time[idx][-1]-time[idx][0])
        Fs = round(len(sample) / duration)
        mtWin = 5.0
        mtStep = 1.0
        stWin = 0.5
        stStep = 0.5
        #print 'FS =',Fs
        [MidTermFeatures, stFeatures] = f1d.mtFeatureExtraction(sample, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))

        if idx == 0:
            emg_features_vectors = MidTermFeatures.copy()
        else:
            emg_features_vectors =  np.concatenate((emg_features_vectors, MidTermFeatures),axis=1)

        #TODO: NOW EACH CSV IS REPRESENTED BY A FV --> NOT THAT

        numOfFeatures = len(stFeatures)
        numOfMidTerm = len(MidTermFeatures)

        #Fx = MidTermFeatures.mean(axis=1)
        #emg_features_vectors.append(Fx)

        #emg_features_vectors.append(Fx)
    return emg_features_vectors






def evaluateClassifier(argv):
    dirName = argv[2]    # path to csv files
    fileList  = sorted(glob.glob(os.path.join(dirName, "*.csv")))

    #data = {}
    #data['user'] = {}
    user = []
    exercise = []
    repetition = []
    time_fatigue = []
    time_nofatigue = []
    time = []
    emg_raw_fatigue = []
    emg_raw_nofatigue = []
    emg_raw = []
    gt_labels = []
    for i, m in enumerate(fileList):


        '''
        if user not in data['user'].keys():
            data['user'][user]={}
            data['user'][user]['exercise'] = {}
        if exercise not in data['user'][user]['exercise'].keys():
            data['user'][user]['exercise'][exercise] = {}
            data['user'][user]['exercise'][exercise]['repetition'] = {}
        if repetition not in data['user'][user]['exercise'][exercise]['repetition'].keys():
            data['user'][user]['exercise'][exercise]['repetition'][repetition] = []
        '''

        with open(m,'r') as f:
            x = f.readlines()
            if not x:
                continue
            time.append( [float(label.split(',')[0]) for label in x ])
            emg_raw.append( [float(label.split(',')[1]) for label in x ])
            gt_labels.append( [int(label.split(',')[2].rstrip()) for label in x ])
        f.close

        #split the sample into the positive and negative classes

        split = gt_labels[-1].index(1)
        emg_raw_fatigue.append(emg_raw[-1][:split])
        emg_raw_nofatigue.append(emg_raw[-1][split:])
        time_fatigue.append(time[-1][:split])
        time_nofatigue.append(time[-1][split:])

        user.append(m.split('/')[-1].split('E')[0][1:])
        exercise.append(m.split('/')[-1].split('R')[0][-1])
        repetition.append(m.split('/')[-1].split('.')[0][-1])


        if argv[-1] == '-s':
            showEMGData(emg_raw[-1],time[-1][-1]-time[-1][0],gt_labels[-1])

        #data['user'][user]['exercise'][exercise]['repetition'][repetition] = [time,emg,gt]

    users_for_training = math.ceil(len(set(user))*0.8)
    #print users_for_training

    #Extract no_fatigue features
    feature_vectors_nofatigue = np.transpose(featureExtraction(emg_raw_nofatigue,time_nofatigue))
    no_fatigue_labels = [0]*feature_vectors_nofatigue.shape[1]

    #Extract fatigue features
    feature_vectors_fatigue = np.transpose(featureExtraction(emg_raw_fatigue,time_fatigue))
    fatigue_labels = [1]*feature_vectors_fatigue.shape[1]


    #Collect all features
    #featuresAll = list(np.transpose(np.concatenate((feature_vectors_nofatigue,feature_vectors_fatigue),axis=1)))
    featuresAll = []
    featuresAll.append(feature_vectors_nofatigue)
    featuresAll.append(feature_vectors_fatigue)
    labelsAll = ['0','1'] # 0:NofFtigue, 1:Fatigue

    #Normilize features
    (featuresAll, MEAN, STD) = aT.normalizeFeatures(featuresAll)

    clf = argv[3][1:]
    params = argv[4]
    #bestParam = aT.evaluateclassifier(featuresAll, labelsAll, 1000, clf , [0.05, 0.1, 0.5], 0, perTrain=0.80)
    bestParam = aT.evaluateclassifier(featuresAll, labelsAll, 1000, clf , params, 0, perTrain=0.80)


    MEAN = MEAN.tolist()
    STD = STD.tolist()

    # STEP C: Save the classifier to file
    Classifier = aT.trainSVM(featuresAll, bestParam)
    modelName = clf +'_'+str(bestParam)
    with open(modelName, 'wb') as fid:  # save to file
        cPickle.dump(Classifier, fid)
    fo = open(modelName + "MEANS", "wb")
    cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(labelsAll, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()



    print 'Training of',clf,'completed'






if __name__ == '__main__':
    if sys.argv[1] == "-c":
        evaluateClassifier(sys.argv)

