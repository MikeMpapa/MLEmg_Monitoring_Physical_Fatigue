from EMG_training_new import evaluateClassifier
from EMG_training_new import featureExtraction
import os
import glob
import numpy
from pyAudioAnalysis import audioTrainTest as aT
from scipy.signal import medfilt




def classifySingleFile(fileName,clf_name,model,MEAN,STD,classNames,filter):
    emg_raw = []
    time=[]
    gt_labels = []
    CM_file = numpy.zeros((len(classNames), len(classNames)))

    with open(fileName, 'r') as f:
        x = f.readlines()
        if  not x:
            return CM_file
        time.append([float(label.split(',')[0]) for label in x])
        emg_raw.append([float(label.split(',')[1]) for label in x])
        gt_labels.append([int(label.split(',')[2].rstrip()) for label in x])
    f.close

    ############
    '''
    mtWin = 5.0
    mtStep = 1.0
    stWin = 0.5
    stStep = 0.5
    mtWinFs = round(mtWin * Fs)
    mtStepFs = round(mtStep * Fs)
    stWinFs = round(Fs * stWin)
    stStepFs = round(Fs * stStep)
    mtWinRatio = int(round(mtWinFs / stStepFs))
    mtStepRatio = int(round(mtStepFs / stStepFs))
    '''
    ##################


    fVs,labels = featureExtraction(emg_raw[0],time[0],gt_labels[0])
    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)


    predictions = []
    for i in range(fVs.shape[1]):
        fV = fVs[:,i]
        fV = (fV - MEAN) / STD
        [Result, P] = aT.classifierWrapper(model, clf_name, fV)  # classification
        predictions.append(Result)

    if filter:
        predictions = medfilt(predictions,13)

    for idx,p in enumerate(predictions):
        CM_file[int(labels[idx]),int(p)] += 1

    print 'Classification Results for file:',fileName
    print CM_file
    print

    return CM_file







def classifyDir(dirName,clf_name,trained_classifier,classNames,MEAN,STD,filter):
    fileList = sorted(glob.glob(os.path.join(dirName, "*.csv")))

    CM = numpy.zeros((len(classNames), len(classNames)))

    for idx, file in enumerate(fileList):
        result = classifySingleFile(file,clf_name,trained_classifier,MEAN,STD,classNames,filter)
        CM += result

    return CM



def computeEvalMetrics(CM,clf_name,bestparam,fold_id):

    CM = CM + 0.0000000010

    Rec = numpy.zeros((CM.shape[0],))
    Pre = numpy.zeros((CM.shape[0],))

    for ci in range(CM.shape[0]):
        Rec[ci] = CM[ci, ci] / numpy.sum(CM[ci, :])
        Pre[ci] = CM[ci, ci] / numpy.sum(CM[:, ci])
    F1 = 2 * Rec * Pre / (Rec + Pre)
    print 'Total CM'
    print CM
    print 'Pre:',Pre,'- AVG Pre:',numpy.mean(Pre)
    print 'Rec:',Rec,'- AVG Rec:',numpy.mean(Rec)
    print 'F1:',F1,'- AVG F1:',numpy.mean(F1)
    numpy.save(clf_name +"_"+fold_id+"_"+str(bestparam) +"_results.npy", CM)




def EMG_Train_Test(evaluation_path,classifier,params,filter):

    folds = sorted(glob.glob(os.path.join(evaluation_path, "fold*")))
    CM_total = None
    for idx,fold in enumerate(folds):
        #Train
        #clf_name,trained_classifier,labelsAll,MEAN,STD = evaluateClassifier([None, '-c', fold+'/train', '-extratrees', [10, 25, 50, 100, 200, 500, 1000], False])
        clf_name,trained_classifier,labelsAll,MEAN,STD,bestParam = evaluateClassifier([None, '-c', fold+'/train', classifier, params, False])
        #Test
        CM = classifyDir(fold+'/test', clf_name, trained_classifier, labelsAll, MEAN, STD,filter)
        if idx==0:
            CM_total = CM
        else:
            CM_total += CM
    computeEvalMetrics(CM_total,clf_name,bestParam,fold.split('/')[-1])






if __name__ == '__main__':

    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-svm',[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-svm_rbf',[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-knn',[1, 3, 5, 7, 9, 11, 13, 15]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-randomforest',[10, 25, 50, 100,200,500,1000]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-gradientboosting',[10, 25, 50, 100,200,500,1000]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-extratrees',[10, 25, 50, 100,200,500,1000],False])

    #EMG_Train_Test('EMGraw_Data_Study1', '-svm',  [0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0],False)
    #EMG_Train_Test('EMGmedian_Data_Study1', '-svm_rbf', [0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0], False)
    EMG_Train_Test('../..//Recognise_Fatigue/src/EMG_Data_Study1', '-knn', [1],False)
    #EMG_Train_Test('EMGraw_Data_Study1', '-randomforest',[10, 25, 50, 100,200,500,1000], False)
    #EMG_Train_Test('../..//Recognise_Fatigue/src/EMG_Data_Study2_System1', '-gradientboosting',[100], True)
    #EMG_Train_Test('EMGraw_Data_Study1', '-extratrees', [10, 25, 50, 100,200,500,1000], False)


    '''
    clfs = ['-svm','-svm_rbf','-knn','-randomforest','-gradientboosting','-extratrees']
    params = [[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0],[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0],[1, 3, 5, 7, 9, 11, 13, 15],[10, 25, 50, 100,200,500,1000],[10, 25, 50, 100,200,500,1000],[10, 25, 50, 100,200,500,1000]]
    for i in range(len(clfs)):
        EMG_Train_Test('EMG_Data_Study1',clfs[i],params[i],False)
        
    '''



