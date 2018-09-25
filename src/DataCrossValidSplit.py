import sys
import glob
import os.path
import numpy
from shutil import copyfile

def split(datapath):

    split_path = 'EMG_Data_'+datapath.split('/')[-3]+'_'+datapath.split('/')[-1]
    if not os.path.exists(split_path):
        os.mkdir(split_path)

    test_ratio = 0.2
    Folds = 10

    fileList  = sorted(glob.glob(os.path.join(datapath, "*.csv")))
    numOf_test_samples = test_ratio*round(len(fileList))

    for fold in range(Folds):
        fold_path = ('/').join((split_path,'fold'+str(fold)))
        train_path = ('/').join((fold_path,'train'))
        test_path = ('/').join((fold_path,'test'))

        if not os.path.exists(fold_path):
            os.mkdir(fold_path)
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        test_samples =  numpy.random.permutation(len(fileList))[:int(numOf_test_samples)]


        for idx,file in enumerate(fileList):
            if idx in test_samples:
                copy_path = ('/').join((test_path, file.split('/')[-1]))
                copyfile(file, copy_path)
            else:
                copy_path = ('/').join((train_path,file.split('/')[-1]))
                copyfile(file, copy_path)












if __name__ == '__main__':
    split(sys.argv[1])
