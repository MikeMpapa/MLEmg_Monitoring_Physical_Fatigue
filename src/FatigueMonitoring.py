from EMG_training import evaluateClassifier


if __name__ == '__main__':

    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-svm',[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-svm_rbf',[0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-knn',[1, 3, 5, 7, 9, 11, 13, 15]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-randomforest',[10, 25, 50, 100,200,500,1000]])
    #evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-gradientboosting',[10, 25, 50, 100,200,500,1000]])
    evaluateClassifier([None,'-c','../Fatigue_Data/Study1/EMG', '-extratrees',[10, 25, 50, 100,200,500,1000]])



