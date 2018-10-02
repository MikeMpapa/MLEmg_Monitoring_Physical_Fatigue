import numpy
from scipy.fftpack import fft
import sys

eps = 0.00000001



def mtFeatureExtraction(signal,Fs, mtWin, mtStep, stWin, stStep):
    """
    Mid-term feature extraction
    """

    mtWinRatio = int(round(mtWin / stStep))
    mtStepRatio = int(round(mtStep / stStep))

    stFeatures = stFeatureExtraction2(signal,Fs, stWin, stStep)

    numOfFeatures = len(stFeatures)
    numOfStatistics = 6

    mtFeatures = []
    # for i in range(numOfStatistics * numOfFeatures + 1):
    for i in range(numOfStatistics * numOfFeatures):
        mtFeatures.append([])

    for i in range(numOfFeatures):  # for each of the short-term features:
        curPos = 0
        N = len(stFeatures[i])
        while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curStFeatures = stFeatures[i][N1:N2]

            mtFeatures[i].append(numpy.mean(curStFeatures))
            mtFeatures[i + numOfFeatures].append(numpy.std(curStFeatures))
            mtFeatures[i + 2 * numOfFeatures].append(numpy.max(curStFeatures))
            mtFeatures[i + 3 * numOfFeatures].append(numpy.min(curStFeatures))
            lower = numpy.sort(curStFeatures)[0:int(curStFeatures.shape[0] / 3)]
            upper = numpy.sort(curStFeatures)[-int(curStFeatures.shape[0] / 3)::]
            if lower.shape[0] > 0:
                mtFeatures[i + 4 * numOfFeatures].append(numpy.mean(lower))
            else:
                mtFeatures[i + 4 * numOfFeatures].append(numpy.mean(curStFeatures))
            if upper.shape[0] > 0:
                mtFeatures[i + 5 * numOfFeatures].append(numpy.mean(upper))
            else:
                mtFeatures[i + 5 * numOfFeatures].append(numpy.mean(curStFeatures))
            '''                
            if lower.shape[0]>0:
                mtFeatures[i+6*numOfFeatures].append(numpy.mean(lower))
            else:
                mtFeatures[i+6*numOfFeatures].append(numpy.mean(curStFeatures))
            if upper.shape[0]>0:
                mtFeatures[i+7*numOfFeatures].append(numpy.mean(upper))
            else:
                mtFeatures[i+7*numOfFeatures].append(numpy.mean(curStFeatures))
            '''
            curPos += mtStepRatio

    return numpy.array(mtFeatures), stFeatures




def stFeatureExtraction2(signal,Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    # signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC)

    N = len(signal)  # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    Features_per_window = 12
    numOfTimeSpectralFeatures = Features_per_window * 2 # each fv cointains 14 features from the current window + 14 features from the previous window
    numOfDeltaFeatures = Features_per_window

    totalNumOfFeatures = numOfTimeSpectralFeatures

    stFeatures = []
    stFeaturesDelta = []


    prevFV = numpy.zeros((totalNumOfFeatures, 1))
    while (curPos + Win - 1 < N):  # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos + Win]  # get current window
        curPos = curPos + Step  # update window position
        X = abs(fft(x))  # get fft magnitude
        X = X[0:nFFT]  # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()  # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((totalNumOfFeatures, 1))
        curFVdelta = numpy.zeros((numOfDeltaFeatures, 1))

        # curFV[0] = stZCR(x)                              # zero crossing rate
        # curFV[1] = stEnergy(x)                           # short-term energy
        # curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        #---- TSF ----
        '''
        curFV[0] = numpy.mean(x)
        curFV[1] = numpy.std(x)
        curFV[2] = numpy.median(x)
        curFV[3] = stZCR(x)
        curFV[4] = numpy.max(x)
        curFV[5] = numpy.min(x)
        curFV[6] = numpy.max(numpy.abs(x))
        curFV[7] = numpy.min(numpy.abs(x))
        curFV[8] = stEnergyEntropy(x)  # short-term entropy of energy
        [curFV[9], curFV[10]] = stSpectralCentroidAndSpread(X, Fs)  # spectral centroid and spread
        curFV[11] = stSpectralEntropy(X)  # spectral entropy
        curFV[12] = stSpectralFlux(X, Xprev)  # spectral flux
        curFV[13] = stSpectralRollOff(X, 0.90, Fs)  # spectral rolloff
        curFV[14] = numpy.median(X)
        '''
        #--- SF ------
        curFV[0] = stSpectralEntropy(X)  # spectral entropy
        curFV[1] = stSpectralFlux(X, Xprev)  # spectral flux
        curFV[2] = stSpectralRollOff(X, 0.90, Fs)  # spectral rolloff
        curFV[3] = numpy.median(X)
        [curFV[4], curFV[5]] = stSpectralCentroidAndSpread(X, Fs)  # spectral centroid and spread
        curFV[6] = numpy.mean(X)
        curFV[7] = numpy.min(X)
        curFV[8] = numpy.max(X)
        curFV[9] = numpy.std(X)
        curFV[10] = stMADev(X)
        curFV[11] = WAMP(x)




        #curFV[6] = stZCR(x)



        #------DELTAS-------#
        # TODO: TEST DELTA
        if countFrames > 1:
            curFV[numOfTimeSpectralFeatures / 2::] = curFV[0:numOfTimeSpectralFeatures / 2] - prevFV[0:numOfTimeSpectralFeatures / 2]
            curFVdelta = curFV[0:numOfTimeSpectralFeatures / 2] - prevFV[0:numOfTimeSpectralFeatures / 2]
        else:
            curFV[numOfTimeSpectralFeatures / 2::] = curFV[0:numOfTimeSpectralFeatures / 2]
            curFVdelta = curFV[0:numOfTimeSpectralFeatures / 2]


        stFeatures.append(curFV)
        stFeaturesDelta.append(curFVdelta)

        prevFV = curFV.copy()
        Xprev = X.copy()

    stFeatures = numpy.concatenate(stFeatures, 1)
    stFeaturesDelta = numpy.concatenate(stFeaturesDelta, 1)
   # print"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
   # print"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    #return stFeaturesDelta
    #sys.exit()
    return stFeatures


def WAMP(x):
    wamp = []
    for i in range(len(x)):
        if i >0:
             if abs(x[i] - x[i-1])>0.0001 : wamp.append(1)
    swamp = sum(wamp)
    #print swamp

    return swamp



def stMADev(X):
    """Compute the Mean absolute deviation"""
    return numpy.mean(abs(X - numpy.mean(X)))



def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count - 1.0))



def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)  # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
        frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy



def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs / (2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)



def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)  # number of frame samples
    Eol = numpy.sum(X ** 2)  # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))  # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)  # compute spectral sub-energies
    En = -numpy.sum(s * numpy.log2(s + eps))  # compute spectral entropy

    return En



def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(Xprev + eps)
    F = numpy.sum((X / sumX - Xprev / sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c * totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)



def showFileAccelerometer(data,duration):
    Fs = round(len(data) / float(duration))
    T = numpy.arange(0, len(data) / float(Fs), 1.0 / Fs)
    plt.plot(T, data)
    plt.show()
