from EMG_training import evaluateClassifier
from EMG_training import featureExtraction
from EMG_training import Classify
import os
import glob
import numpy
from pyAudioAnalysis import audioTrainTest as aT
from scipy.signal import medfilt
import Classification as clf
from copy import copy
from keras.preprocessing import image


def classifySingleFile(fileName,clf_name,model,MEAN,STD,classNames,filter):
    '''
    Classify a csv using mid-term windows as samples
    :param fileName: file to classify
    :param clf_name: classifier name ie 'svm
    :param model: trained model
    :param MEAN: mean of training data
    :param STD: std of training data
    :param classNames: list of unique class-names
    :param filter: either to apply midfiltering or not
    :return: Confusion matrix of classified file - mid term windows are the samples
    '''

    emg_raw = []
    time=[]
    gt_labels = []
    CM_file = numpy.zeros((len(classNames), len(classNames)))

    #read the data
    with open(fileName, 'r') as f:
        x = f.readlines()
        if  not x:
            return CM_file
        time.append([float(label.split(',')[0]) for label in x])
        emg_raw.append([float(label.split(',')[1]) for label in x])
        gt_labels.append([int(label.split(',')[2].rstrip()) for label in x])
    f.close
    # extract the features and mid-term labels
    fVs,labels = featureExtraction(emg_raw[0],time[0],gt_labels[0],2,1,0.25,0.25)

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)
    #classify mid-term windows extracted from test-file
    predictions = []
    for i in range(fVs.shape[1]):
        fV = fVs[:,i]
        fV = (fV - MEAN) / STD
        [Result, P] = aT.classifierWrapper(model, clf_name, fV)  # classification
        predictions.append(Result)
    #perform median filtering
    if filter:
        predictions = medfilt(predictions,13)
    # compute confusion matrix
    for idx,p in enumerate(predictions):
        CM_file[int(labels[idx]),int(p)] += 1

    print 'Classification Results for file:',fileName
    print CM_file
    print
    return CM_file



def classifyDir(dirName,clf_name,trained_classifier,classNames,MEAN,STD,filter):
    '''
    Classify all csv files in a dir
    :param dirName: dir where the csv data are
    :param clf_name: name of classifier ie 'svm'
    :param trained_classifier: trained model
    :param classNames: list of unique class name
    :param MEAN: mean of training features
    :param STD: std of training features
    :param filter: if to apply median filtering
    :return: returns confusion matrix across all files in dir
    '''

    fileList = sorted(glob.glob(os.path.join(dirName, "*.csv")))
    CM = numpy.zeros((len(classNames), len(classNames)))
    for idx, file in enumerate(fileList):
        result = classifySingleFile(file,clf_name,trained_classifier,MEAN,STD,classNames,filter)
        CM += result
    return CM



def computeEvalMetrics(CM,filename='results',save=False):
    '''
    Compute Precision, Recall & F1 given a Confusion Matrix
    :param CM: confusion matrix
    :param filename: name of output file - only if save is true
    :param save: if results will be stored
    :return: stores confusion matrix if save==True
    '''
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
    if save:
        numpy.save(filename +"_results.npy", CM)


def EMG_Train_Test(evaluation_path,classifier,params,filter):
    '''
    Evaluate accross the files as splited into train and test given a path --- task-dependent function  !pyAudioAnalysis dependent!

    :param evaluation_path: path where the deferent folds are
    :param classifier: classifier name ie. 'svm'
    :param params: params to cross validate and fine-tune classifier
    :param filter: if median filtering needs to be applyied in predicted signal
    :return: None
    '''

    folds = sorted(glob.glob(os.path.join(evaluation_path, "fold*")))
    CM_total = None
    num_of_folds=0
    for idx,fold in enumerate(folds):
        #Train
        clf_name,trained_classifier,labelsAll,MEAN,STD,bestParam = evaluateClassifier([None, '-c', fold+'/train', classifier, params, False])
        #Test
        CM = classifyDir(fold+'/test', clf_name, trained_classifier, labelsAll, MEAN, STD,filter)
        if idx==0:
            CM_total = CM
        else:
            CM_total += CM
        if idx == num_of_folds:
            break
    f = ('_').join((clf_name,str(bestParam),str(fold.split('/')[-1])))
    computeEvalMetrics(CM_total,f)



def postProcessing(predictions,filter=True):
    '''
    Performs post-processing of the raw classifier predictions based on a history of 2 previous windows of length w
    :param predictions: list of predicted labels
    :return: updated predictions after post processing
    '''

    if filter:
        predictions = list(medfilt(predictions,3))

    w = 3 #window length
    step = 1#int(w/3) #window step
    thresh = 0.6 #confidence threshold

    prev3 = None
    prev2 = None #history window 1
    prev1 = None #history window 2
    N=0
    N1 = w
    count = 0
    complete = False #if post processing didnt capture fatigue return predictions full of zeros ie. NO-FATIGUE
    while True:
        count += 1
        if N1 > len(predictions):
            N1 = len(predictions)+1
            complete = True
        x = predictions[N:N1]

        if prev2 != None:
            if x.count(1)/float(w)>=thresh and prev1.count(1)/float(w)>=thresh and prev2.count(1)/float(w)>=thresh :
                post_predictions = [0]*count+[1]*len(predictions[count:])
                break

        prev3 = copy(prev2)
        prev2 = copy(prev1)
        prev1 = copy(x)
        N += step
        N1 += step

        if complete:
            print 'NONE'
            post_predictions = [0]*len(predictions)
            break
    return post_predictions





def UserEvaluation(dirName,classifier,c_param):
    '''
    Cross-Validate across users: N-1 users for training and 1 user for testing
    :param dirName: path to extracted mid-term feature matrices
    :param classifier: classifier name ie. 'svm'
    :param c_param: cclassifier parameter
    :return: None
    '''
    fileList  = sorted(glob.glob(os.path.join(dirName, "*NF.npz")))
    user_ids =  sorted(list(set(map(lambda x:x.split('/')[-1].split('_')[0],fileList))))
    CM = numpy.zeros((2, 2))
    CM_post = numpy.zeros((2, 2))
    for uid,user in enumerate(user_ids):
        print "Test on user:", user
        train_data_F, train_data_NF, test_data_F, test_data_NF, test_ids = DataCollect(fileList, user,'u')
        CM_user, CM_user_post, predictions, post_predictions =  GroupClassification([train_data_NF,train_data_F],[test_data_NF,test_data_F],classifier,c_param,test_ids, dirName)
        CM += CM_user
        CM_post += CM_user_post
    PrintResults('-', "TOTAL RESULTS ACROSS ALL USERS", CM, CM_post)



def ExerciseEvaluation(dirName,classifier,c_param):
    '''
    Cross-Validate across exercises: N-1 exercises for training and 1 exercise for testing
    :param dirName: path to extracted mid-term feature matrices
    :param classifier: classifier name ie. 'svm'
    :param c_param: cclassifier parameter
    :return: None
    '''
    fileList  = sorted(glob.glob(os.path.join(dirName, "*NF.npz")))
    exercise_ids =  sorted(list(set(map(lambda x:x.split('/')[-1].split('_')[1],fileList))))
    CM = numpy.zeros((2, 2))
    CM_post = numpy.zeros((2, 2))
    for eid,exercise in enumerate(exercise_ids):
        print "Test on exercise:", exercise
        train_data_F, train_data_NF, test_data_F, test_data_NF, test_ids = DataCollect(fileList, exercise,'e')
        CM_exercise, CM_exercise_post, predictions, post_predictions = GroupClassification([train_data_NF, train_data_F],[test_data_NF, test_data_F], classifier, c_param,test_ids, dirName)
        CM += CM_exercise
        CM_post += CM_exercise_post
    PrintResults('-', "TOTAL RESULTS ACROSS ALL EXERCISES", CM, CM_post)



def SingleUserEvaluation(dirName,classifier,c_param):
    '''
    Cross-Validate across files of a single user: N-1 files of a user for training and 1 user file for testing
    :param dirName: path to extracted mid-term feature matrices
    :param classifier: classifier name ie. 'svm'
    :param c_param: cclassifier parameter
    :return: None
    '''
    fileList  = sorted(glob.glob(os.path.join(dirName, "*NF.npz")))
    user_ids =  sorted(list(set(map(lambda x:x.split('/')[-1].split('_')[0],fileList))))
    CM_avg= numpy.zeros((2, 2))
    CM_avg_post = numpy.zeros((2, 2))
    for uid,user in enumerate(user_ids):
        print "Test on user:", user
        CM = numpy.zeros((2, 2))
        CM_post = numpy.zeros((2, 2))
        user_files = filter(lambda x:x.split('/')[-1].split('_')[0] == user,fileList)
        for file_test in user_files:
            train_data_F, train_data_NF, test_data_F, test_data_NF, test_ids = DataCollect(user_files,file_test,'')
            CM_user, CM_user_post, predictions, post_predictions = GroupClassification([train_data_NF, train_data_F], [test_data_NF, test_data_F], classifier, c_param,test_ids,dirName)
            CM += CM_user
            CM_post += CM_user_post
        CM_avg += CM
        CM_avg_post +=CM_post
        PrintResults('-',"TOTAL RESULTS ACROSS FILES OF USER "+user,CM,CM_post)
    PrintResults('+',"TOTAL RESULTS ACROSS ALL USERS",CM_avg,CM_avg_post)




def SingleExerciseEvaluation(dirName,classifier,c_param):
    '''
    Cross-Validate across files of a single exercise: N-1 files of an exercise for training and 1 exercise file for testing
    :param dirName: path to extracted mid-term feature matrices
    :param classifier: classifier name ie. 'svm'
    :param c_param: cclassifier parameter
    :return: None
    '''
    fileList  = sorted(glob.glob(os.path.join(dirName, "*NF.npz")))
    exercise_ids =  sorted(list(set(map(lambda x:x.split('/')[-1].split('_')[1],fileList))))
    CM_avg= numpy.zeros((2, 2))
    CM_avg_post = numpy.zeros((2, 2))
    for eid,exercise in enumerate(exercise_ids):
        print "Test on exerciser:", exercise
        CM = numpy.zeros((2, 2))
        CM_post = numpy.zeros((2, 2))
        exercise_files = filter(lambda x:x.split('/')[-1].split('_')[1] == exercise,fileList)
        for file_test in exercise_files:
            train_data_F, train_data_NF, test_data_F, test_data_NF, test_ids = DataCollect(exercise_files,file_test,'')
            CM_exercise, CM_exercise_post, predictions, post_predictions = GroupClassification( [train_data_NF, train_data_F], [test_data_NF, test_data_F], classifier, c_param,test_ids,dirName)
            CM += CM_exercise
            CM_post += CM_exercise_post
        CM_avg += CM
        CM_avg_post +=CM_post
        PrintResults('-',"TOTAL RESULTS ACROSS FILES OF EXERCISE "+exercise,CM,CM_post)
    PrintResults('+',"TOTAL RESULTS ACROSS ALL EXERCISES",CM_avg,CM_avg_post)



def CrossStudyEvaluation(t_path,tst_path,classifier,c_param):
    '''
    Train classifier using data from one dir and evaluate using data from another dir
    :param t_path: path to training data
    :param tst_path: path to test data
    :param classifier: classifier name ie. 'svm'
    :param c_param: cclassifier parameter
    :return: None
    '''
    fileList  = sorted(glob.glob(os.path.join(t_path, "*NF.npz"))) + sorted(glob.glob(os.path.join(tst_path, "*NF.npz")))
    train_data_F, train_data_NF, test_data_F, test_data_NF, test_ids = DataCollect(fileList, tst_path, 'cross_study')
    CM, CM_post, predictions, post_predictions = GroupClassification([train_data_NF, train_data_F], [test_data_NF, test_data_F], classifier,c_param, test_ids, tst_path)
    PrintResults('-', "TOTAL RESULTS ACROSS STUDIES", CM, CM_post)


def GroupClassification(train,test,classifier,param,test_ids,eval_source):
    '''
    Perform classification and post processing given a set of data
    :param train: train[0]--> list of NO-FATIGUE samples, train[1]--> list of FATIGUE-SAMPLES
    :param test: test[0]--> list of NO-FATIGUE samples, test[1]--> list of FATIGUE-SAMPLES
    :param classifier: classifier name ie. 'svm'
    :param param: classifier parameter
    :return: Original Confusion Matrix, Confusion Matrix after post-processing,original predicted labels, predicted labels after post-processing
    '''
    CM = numpy.zeros((2, 2))
    CM_post = numpy.zeros((2, 2))
    # from lists to matrices
    trNF = numpy.concatenate(train[0])
    trF = numpy.concatenate(train[1])

    # normalize train features - 0mean -1std
    features_norm, MEAN, STD = clf.normalizeFeatures([trNF, trF])
    #train the classifier
    model = Classify(classifier, features_norm, param)

    # TEST
    for recording in range(len(test[0])):
        predictions = []
        probs = []
        test_labels = [0] * test[0][recording].shape[0] + [1] * test[1][recording].shape[0]
        test_recording_fVs = numpy.concatenate((test[0][recording], test[1][recording]))

        for i in range(test_recording_fVs.shape[0]):
            fV = test_recording_fVs[i, :]
            fV = (fV - MEAN) / STD
            [Result, P] = clf.classifierWrapper(model, classifier, fV)  # classification
            probs.append(numpy.max(P))
            predictions.append(Result)

        for idx, gtlabel in enumerate(test_labels):
            CM[int(gtlabel), int(predictions[idx])] += 1

        post_predictions = postProcessing(predictions)
        for idx, gtlabel in enumerate(test_labels):
            CM_post[int(gtlabel), int(post_predictions[idx])] += 1
        CompareToInitialStudy(post_predictions,test_ids[recording],eval_source)
    return CM,CM_post,predictions,post_predictions



def DataCollect(filelist,test_id,totest):
    '''
    Collects the data given the experiment: totest=='u' --> user cross-validation,totest=='ue' --> exercise cross-validation,
    else single user or single exercise cross validation based on filelist parameter
    :param filelist:
    :param test_id:
    :param totest:
    :return:
    '''
    test_data_F = []
    test_data_NF = []
    train_data_F = []
    train_data_NF = []
    test_ids = []
    for file in filelist:

        if totest == 'u':
            cur_id = file.split('/')[-1].split('_')[0]
        elif totest == 'e':
            cur_id = file.split('/')[-1].split('_')[1]
        elif totest == 'cross_study':
            cur_id = ('/').join((file.split('/')[:-1]))
        else:
            cur_id = file

        if cur_id == test_id :
            test_ids.append( file.split('/')[-1].replace('_NF',''))
            with numpy.load(file) as data:
                test_data_NF.append(data[data.keys()[0]])
            data.close()
            with numpy.load(file.replace('NF','F')) as data:
                test_data_F.append(data[data.keys()[0]])
            data.close()
        else:
            with numpy.load(file) as data:
                train_data_NF.append(data[data.keys()[0]])
            data.close()
            with numpy.load(file.replace('NF', 'F')) as data:
                train_data_F.append(data[data.keys()[0]])
            data.close()
    return train_data_F,train_data_NF, test_data_F,test_data_NF,test_ids


def PrintResults(s,str,CM,CM_post):
    '''
    Prints results before and after post-processing
    '''
    print  s*20
    print str
    print "NO-POSTPROCESSING"
    computeEvalMetrics(CM)
    print "POSTPROCESSING"
    computeEvalMetrics(CM_post)



def CompareToInitialStudy(predictions,file,eval_source):

    if 'Study1' in eval_source:
        eval_source = 'Study1'
    elif 'Study2.1' in eval_source:
        eval_source = 'Study2.1'
    elif 'Study2.2' in eval_source:
        eval_source = 'Study2.2'

    if not os.path.exists("../experiment_comparison"):
        os.mkdir("../experiment_comparison")

    augmented_predictions = []

    with numpy.load('../original_labels_'+eval_source+'/'+file) as labels:
        gtlabels = list(labels[labels.keys()[0]])
    with numpy.load('../original_times_'+eval_source+'/'+file) as times:
        gttimes = list(times[times.keys()[0]])
    Fs = int(round(len(gttimes) / float(gttimes[-1]-gttimes[0])))
    for idx,pred_label in enumerate(predictions):
        augmented_predictions = augmented_predictions + [pred_label]*Fs
    augmented_predictions = augmented_predictions[:len(gtlabels)]
    zipdata = zip(gttimes, gtlabels, augmented_predictions)
    with open("../experiment_comparison/"+file.replace('npz','csv'),'w') as f:
        for p in zipdata:
            f.write(('\t').join((str(p[0]),str(p[1]),str(p[2]))))
            f.write('\n')
    f.close()




def CNN_classification(test_id,valid_id):
    input_size = 100

    def LoadData(fileList_NF,fileList_F):
        data = []
        labels = []
        for im_path in fileList_NF:
            img = numpy.asarray(image.load_img(im_path, target_size=(input_size, input_size)))
            data.append(img)
            labels.append('0')
        for im_path in fileList_F:
            img = numpy.asarray( image.load_img(im_path, target_size=(input_size, input_size)),dtype='float64')
            data.append(img)
            labels.append('1')
        return numpy.asarray(data),numpy.asarray(labels)


    fileList_NF = sorted(glob.glob('medfilt5_label_0/*.png'))
    fileList_F = sorted(glob.glob('medfilt5_label_1/*.png'))


    testfileList_NF = filter(lambda x: x.split('/')[-1].split('E')[0] == 'U'+str(test_id), fileList_NF)
    testfileList_F = filter(lambda x: x.split('/')[-1].split('E')[0] == 'U'+str(test_id), fileList_F)
    validfileList_NF = filter(lambda x: x.split('/')[-1].split('E')[0] == 'U'+str(valid_id), fileList_NF)
    validfileList_F = filter(lambda x: x.split('/')[-1].split('E')[0] == 'U'+str(valid_id), fileList_F)
    trainfileList_NF = filter(lambda x: x.split('/')[-1].split('E')[0] != 'U' + str(test_id) and x.split('/')[-1].split('E')[0] != 'U' + str(valid_id), fileList_NF)
    trainfileList_F = filter(lambda x: x.split('/')[-1].split('E')[0] != 'U' + str(test_id) and x.split('/')[-1].split('E')[0] != 'U' + str(valid_id), fileList_F)

    train_data,train_labels = LoadData(trainfileList_NF,trainfileList_F)
    test_data,test_labels = LoadData(testfileList_NF,testfileList_F)
    valid_data,valid_labels = LoadData(validfileList_NF,validfileList_F)
    print train_data.shape,test_data.shape,valid_data.shape

    clf.CNNclassifier(train_data,train_labels,valid_data,valid_labels,input_size)





if __name__ == '__main__':
    '''
    NOTES
    median filter -11
    fixed post processing
    '''

    #---------USER CROSS-VALIDATION-----------#
    #UserEvaluation("../Study1_medfilt11_EMG",'knn',5)
    #UserEvaluation("../Study1_medfilt11_EMG",'svm_rbf',100)
    #UserEvaluation("../Study1_medfilt11_EMG",'randomforest',500)
    #UserEvaluation("../Study1_medfilt11_EMG",'gradientboosting',1000)
    #UserEvaluation("../Study1_medfilt11_EMG",'extratrees',500)
    #UserEvaluation("../Study1_medfilt11_EMG",'svm',100)


    #---------EXERCISE CROSS-VALIDATION-----------#
    #ExerciseEvaluation("../Study1_medfilt11_EMG",'svm',100)
    #ExerciseEvaluation("../Study1_medfilt11_EMG",'svm_rbf',100)
    #ExerciseEvaluation("../Study1_medfilt11_EMG",'extratrees',500)
    #ExerciseEvaluation("../Study1_medfilt11_EMG",'randomforest',500)
    #ExerciseEvaluation("../Study1_medfilt11_EMG",'gradientboosting',1000)

    #---------SINGLE USER EVALUATION--------------#
    #SingleUserEvaluation("../Study2.1_EMG",'svm_rbf',10)
   # SingleUserEvaluation("../Study1_medfilt11_EMG", 'gradientboosting', 1000)
    #SingleUserEvaluation("../Study1_medfilt11_EMG",'svm_rbf',100)
    #SingleUserEvaluation("../Study1_medfilt11_EMG",'randomforest',500)
    #SingleUserEvaluation("../Study1_medfilt11_EMG",'gradientboosting',1000)
    #SingleUserEvaluation("../Study1_medfilt11_EMG",'extratrees',500)
    #SingleUserEvaluation("../Study1_medfilt11_EMG",'svm',100)

    # ---------SINGLE EXERCISE EVALUATION--------------#
    SingleExerciseEvaluation("../Study1_medfilt11_EMG", 'gradientboosting', 1000)
    #SingleExerciseEvaluation("../Study1_medfilt11_EMG",'svm_rbf',100)
    #SingleExerciseEvaluation("../Study1_medfilt11_EMG",'randomforest',500)
    #SingleExerciseEvaluation("../Study1_medfilt11_EMG",'gradientboosting',1000)
    #SingleExerciseEvaluation("../Study1_medfilt11_EMG",'extratrees',500)
    #SingleExerciseEvaluation("../Study1_medfilt11_EMG",'svm',100)

    # ---------CROSS STUDY EVALUATION--------------#
    #CrossStudyEvaluation('../Study1_medfilt11_EMG','../Study2.1_medfilt11_EMG','gradientboosting',1000)

    #CNN_classification(1,2)

    '''
    clfs = ['-svm','-svm_rbf','-knn','-randomforest','-gradientboosting','-extratrees']
    params = [[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0],[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0],[1, 3, 5, 7, 9, 11, 13, 15],[10, 25, 50, 100,200,500,1000],[10, 25, 50, 100,200,500,1000],[10, 25, 50, 100,200,500,1000]]
    for i in range(len(clfs)):
        EMG_Train_Test('EMG_Data_Study1',clfs[i],params[i],False)
        
    '''


    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-svm',[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-svm_rbf',[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-knn',[1, 3, 5, 7, 9, 11, 13, 15]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-randomforest',[10, 25, 50, 100,200,500,1000]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-gradientboosting',[10, 25, 50, 100,200,500,1000]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-extratrees',[10, 25, 50, 100,200,500,1000],False])

    #EMG_Train_Test('EMGraw_Data_Study1', '-svm',  [0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0],False)
    #EMG_Train_Test('../..//Recognise_Fatigue/src/EMG_Data_Study2_System1', '-svm_rbf', [0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0], False)
    #EMG_Train_Test('../..//Recognise_Fatigue/src/EMG_Data_Study2_System1', '-knn', [1],False)
    #EMG_Train_Test('EMGraw_Data_Study1', '-randomforest',[10, 25, 50, 100,200,500,1000], False)
    #EMG_Train_Test('../..//Recognise_Fatigue/src/EMG_Data_Study2_System1', '-gradientboosting',[500], False)
    #EMG_Train_Test('EMGraw_Data_Study1', '-extratrees', [10, 25, 50, 100,200,500,1000], False)
