from PyEMD import EMD
from sklearn import svm
from sklearn import preprocessing
from sklearn import utils
import pickle
import numpy as np
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt


def read_eeg_signal_from_file(filename):
    print(filename)
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    p = x.load()
    return p


def get_file_name(number):
    if len(number) == 1:
        filename = "s0"
    else:
        filename = "s"
    filename += number + ".dat"
    return filename


def get_first_difference_of_time_series(IMF):
    N = len(IMF)
    imf_range = range(0, N-1) # da 1 a N-1
    sum = 0
    for n in imf_range:
        sum += np.abs(IMF[n+1] - IMF[n])
    D_t = sum/(N-1)
    return D_t


def get_first_difference_of_phase(IMF):
    z = hilbert(IMF)
    phi = np.unwrap(np.angle(z)) # instantaneous phase ---> phi(n) = arctan(y(n)/x(n))
    N = len(IMF)
    imf_range = range(0, N-1) # da 1 a N-1
    sum = 0
    for n in imf_range:
        # z_n = hilbert(IMFs[n])
        # z_n1 = hilbert(IMFs[n+1])
        # phi_n = np.unwrap(np.angle(z)) # instantaneous phase ---> phi(n) = arctan(y(n)/x(n))
        # phi_n1 = np.unwrap(np.angle(z))
        sum += np.abs(phi[n+1] - phi[n])
    D_p = sum/(N-1)
    return D_p


def get_normalized_energy(orig_sig, IMF):
    N = len(IMF)
    imf_range = range(0, N) # da 1 a N
    sum1 = 0
    sum2 = 0
    for n in imf_range:
        sum1 += pow(IMF[n], 2)
        sum2 += pow(orig_sig[n], 2)
    E_norm = sum1/sum2
    return E_norm


# Estrae dal video un sample di segnale ogni 5 secondi,
# quindi si hanno 672 elementi ogni sample (ognuno dei quali ha 40 canali)
def get_samples(samples, video):
    start = 0
    end = 671
    for n in range(0, 12):
        samples.append(video[start:end])
        start = end+1
        end += 672
    return samples


def get_channels(index, samples):
    channel_fp1 = np.array(samples[index][0])
    channel_fp2 = np.array(samples[index][16])
    channel_f7 = np.array(samples[index][3])
    channel_f8 = np.array(samples[index][20])
    channel_t7 = np.array(samples[index][7])
    channel_t8 = np.array(samples[index][25])
    channel_p7 = np.array(samples[index][11])
    channel_p8 = np.array(samples[index][29])

    channels = np.array([channel_fp1, channel_fp2, channel_f7,
                         channel_f8, channel_t7, channel_t8, channel_p7, channel_p8])
    return channels


def extract_features(channels):
    features = []
    for n in range(0, len(channels)):
        print(channels[n])
        t = np.linspace(0, 1, len(channels[n]))
        IMFs = EMD().emd(channels[n], None, 1)

        ###### Per ogni canale, estrai le 3 features ######

        IMF = IMFs[0]
        D_t = get_first_difference_of_time_series(IMF)
        D_p = get_first_difference_of_phase(IMF)
        E_norm = get_normalized_energy(channels[n], IMF)

        features.append(D_t)
        features.append(D_p)
        features.append(E_norm)
    return features




class Main:
    ###### Leggi il file dell'esperimento ed estrai i due array (labes e data) ######

    filename = "E:\DatasetDEAP\DEAP\physiological recordings\data_preprocessed_python\s01.dat"
    experiment = read_eeg_signal_from_file(filename)
    labels = experiment['labels']
    # print(labels)
    data = experiment['data']
    # print(data)

    ###### Per ogni segnale nel dataset, decomponi il segnale in IMFs ######

    video_0 = data[0]
    # video_n = data[n]

    channel_fp1 = video_0[0]
    channel_fp2 = video_0[16]
    channel_f7 = video_0[3]
    channel_f8 = video_0[20]
    channel_t7 = video_0[7]
    channel_t8 = video_0[25]
    channel_p7 = video_0[11]
    channel_p8 = video_0[29]
    # channel_nm = video_n[m]

    # print("CHANNEL: " + str(channel_fp1))

    channels = np.array([channel_fp1, channel_fp2, channel_f7,
                         channel_f8, channel_t7, channel_f8, channel_p7, channel_p8])

    # signal_000 = channel_fp1[0]
    # channel_nmp = channel_nm[p]

    # print("SIGNAL " + str(signal_000))

    # signal = readEEGSignalFromFile(filename)
    # print(signal)

    # emd = EMD()

    # samples = []
    # for i in range(0, 5):
    #     samples.append(data[i])

    X = []
    Y = []

    samples = []
    for n in range(0, len(data)-1):
        samples = get_samples(samples, data[n])

    print(len(samples))

    # Per ogni sample, estrai le features dagli 8 canali e inserisci il vettore ottenuto in X
    # Inserisci la label relativa al sample in Y
    for m in range(0, len(samples)):
        features = []

        channels = get_channels(m, samples)

        features = extract_features(channels)

        dtype = np.float64

        features = np.array(features, dtype=dtype)

        X.append(features)

    for l in range(0, len(labels)-1):
        valence = 0
        arousal = 0

        if labels[l][0] >= 5:
            valence = 1

        if labels[l][1] >= 5:
            arousal = 1

        for i in range(0, 12):
            Y.append(valence)


    ###### Dai in input le features all'svm ######

    X = np.array(X, dtype=dtype)
    Y = np.array(Y, dtype=dtype)

    print("X: " + str(X))
    print("Y: " + str(Y))

    # libsvm = svm.libsvm
    #
    # libsvm.fit(X, y)
    #
    # libsvm.predict(X, 0, 'rbf', 3, 'auto', 0.0)


    svm = svm.SVC()

    svm.fit(X, Y, None)

    T = []

    predict_samples = get_samples(data[39])

    # Per ogni sample, estrai le features dagli 8 canali e inserisci il vettore ottenuto in X
    # Inserisci la label relativa al sample in Y
    for m in range(0, len(predict_samples)):
        predict_channels = get_channels(m, predict_samples)
        predict_features = extract_features(predict_channels)
        T.append(predict_features)

        dtype = np.float64

        predict_features = np.array(predict_features, dtype=dtype)

        T.append(predict_features)


    T = np.array(T, dtype=dtype)

    prediction = svm.predict(T)

    print("Predicted values: " + str(prediction))

    print("Label: " + labels[39][0])

    # for n in range(0, len(channels)):
    #     print(channels[n])
    #     t = np.linspace(0, 1, len(channels[n]))
    #     IMFs = EMD().emd(channels[n], None, 1)
    #     # N = IMFs.shape[0] + 1
    #     # print(IMFs.shape[0])
    #     position = 1
    #
    #     # print(channel_fp1)
    #     # print(IMFs[0])
    #     # print(IMFs[1])
    #
    #     # # Plot results
    #     # plt.subplot(2, 1, position)
    #     # plt.plot(t, channel_fp1, 'r')
    #     # plt.title("Input signal")
    #     # plt.xlabel("Time [s]")
    #
    #     # for n, imf in enumerate(IMFs):
    #     #     position+=1
    #     #     if n == 1:
    #     #         break
    #     #     plt.subplot(2, 1, position)
    #     #     plt.plot(t, imf, 'g')
    #     #     plt.title("IMF " + str(n + 1))
    #     #     plt.xlabel("Time [s]")
    #
    #     # plt.tight_layout()
    #     # plt.savefig('simple_example')
    #     # plt.show()
    #     # #
    #     # # print(IMFs)
    #     # #
    #     # # print(IMFs[1])
    #
    #     ###### Per ogni canale, estrai le 3 features ######
    #
    #     IMF = IMFs[0]
    #     D_t = getFirstDifferenceOfTimeSeries(IMF)
    #     D_p = getFirstDifferenceOfPhase(IMF)
    #     E_norm = getNormalizedEnergy(channels[n], IMF)
    #
    #     # print(D_t)
    #     # print(D_p)
    #     # print(E_norm)
    #
    #     # features.append(np.array([D_t, D_p, E_norm]))
    #     features.append(D_t)
    #     features.append(D_p)
    #     features.append(E_norm)

    # lab_enc = preprocessing.LabelEncoder()
    # encoded = lab_enc.fit_transform(labels[0])

    # y = np.array(labels[0], dtype=dtype)

    # encoded_labels = []
    #
    # for n in range(0, len(channels)):
    #     encoded_labels.append(encoded)
    #
    # Y = np.array(encoded_labels)

    # valence = 0
    # arousal = 0
    #
    # if labels[0][0] >= 5:
    #     valence = 1
    #
    # if labels[0][1] >= 5:
    #     arousal = 1

    # prova = []
    #
    # for n in range(0, len(channels)):
    #     prova.append(n)
