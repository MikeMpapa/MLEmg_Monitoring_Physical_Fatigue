#!/usr/bin/env python2.7
import scipy.misc
import os
import sys
import numpy as np
import glob
import scipy
from pyAudioAnalysis import audioFeatureExtraction as aF
import matplotlib.patches
from PIL import Image
import cv2
import matplotlib.cm
import scipy.signal as filter


def createSpectrogramFile(x, Fs, fileName, stWin, stStep,label):
    specgramOr, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs * stWin), round(Fs * stStep), False)
    specgramOr = filter.medfilt2d(specgramOr,5)
    save_path = "medfilt5_label_"+label+'/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    specgram = cv2.resize(specgramOr, (227, 227), interpolation=cv2.INTER_LINEAR)
    im1 = Image.fromarray(np.uint8(matplotlib.cm.jet(specgram) * 255))
    scipy.misc.imsave(save_path+fileName, im1)



def main(argv):
    dirName = argv[1]
    types = ('*.csv',)
    filesList = []
    for files in types:
        filesList.extend(glob.glob(os.path.join(dirName, files)))
    filesList = sorted(filesList)

    WIDTH_SEC = 2
    WIDTH_SEC_STEP =0.01
    stWin = 0.040
    stStep = 0.02

    for file in filesList:

        with open(file, 'r') as f:
            data = f.readlines()
            if not data:
                continue
            t = np.asarray([float(label.split(',')[0]) for label in data])
            x = np.asarray([float(label.split(',')[1]) for label in data])
            gt_labels = np.asarray([int(label.split(',')[2].rstrip()) for label in data])
        f.close

        Fs = round(len(data)/(t[-1] - t[0]))
        x = x.astype(float) / x.max()

        N1=0
        N2=int(WIDTH_SEC*Fs)
        done = False
        counter = 0
        while (True):
            counter += 1
            x1 = x[N1:N2]
            label = str(np.where(np.bincount(gt_labels[N1:N2]) == np.max(np.bincount(gt_labels[N1:N2])))[0][0])
            if N2 > x.shape[0]:
                N2 = x.shape[0]
                done = True
            createSpectrogramFile(x1, Fs, file.replace(".csv", "").split('/')[-1]+'_'+str(counter)+'.png', stWin, stStep,label)
            N1 += int(WIDTH_SEC_STEP*Fs)
            N2 += int(WIDTH_SEC_STEP*Fs)
            if done: break


if __name__ == '__main__':
    inputs = sys.argv
    #global inputs
    main(inputs)


