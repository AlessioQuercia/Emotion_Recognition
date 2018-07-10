from PyEMD import EMD
from sklearn import svm
import pickle
import numpy as np
from scipy.signal import hilbert
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
    # phi = np.unwrap(np.angle(z)) # instantaneous phase ---> phi(n) = arctan(y(n)/x(n))
    phi = np.angle(z)
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
# quindi si hanno 672 elementi ogni sample (ognuno dei quali è composto da 40 canali)
def get_samples(samples, video):
    start = 0
    end = 671
    for n in range(0, 12):
        samples.append(video[start:end])
        start = end+1
        end += 672
    return samples


def get_channels(index, data):
    channel_fp1 = np.array(data[index][0])
    channel_fp2 = np.array(data[index][16])
    channel_f7 = np.array(data[index][3])
    channel_f8 = np.array(data[index][20])
    channel_t7 = np.array(data[index][7])
    channel_t8 = np.array(data[index][25])
    channel_p7 = np.array(data[index][11])
    channel_p8 = np.array(data[index][29])

    channels = np.array([channel_fp1, channel_fp2, channel_f7,
                         channel_f8, channel_t7, channel_t8, channel_p7, channel_p8])
    return channels


def extract_features_grouped_for_channel(X, channels):
    samples = []
    features = []
    for n in range(0, 12):
        samples.append([])
        features.append([])
    for n in range(0, len(channels)):
        # if n == 1:
        #     # Solo un canale
        #     break
        start = 0
        end = 672
        for i in range(0, 12):
            signal_sample = np.array(channels[n][start:end])

            samples[i].append(signal_sample)

            IMFs = EMD().emd(signal_sample, None, 1)

            # plot_IMFs(signal_sample, IMFs)

            #
            # print(IMFs)
            #
            # print(IMFs[1])

            ###### Per ogni canale, estrai le 3 features ######

            IMF = IMFs[0]
            D_t = get_first_difference_of_time_series(IMF)
            D_p = get_first_difference_of_phase(IMF)
            E_norm = get_normalized_energy(signal_sample, IMF)

            features[i].append(D_t)
            features[i].append(D_p)
            features[i].append(E_norm)

            start = end
            end += 672

    # print(len(samples))

    samples = np.array(samples)

    features = np.array(features)

    for f in features:
        X.append(np.array(f))

    # for s in range(0, len(samples)):
    #     features = []
    #     IMFs = EMD().emd(samples[s], None, 1)
    #
    #     ###### Per ogni canale, estrai le 3 features ######
    #
    #     IMF = IMFs[0]
    #     D_t = get_first_difference_of_time_series(IMF)
    #     D_p = get_first_difference_of_phase(IMF)
    #     E_norm = get_normalized_energy(samples[s], IMF)
    #
    #     features.append(D_t)
    #     features.append(D_p)
    #     features.append(E_norm)
    #
    #     X.append(np.array(features))

    return X

def extract_features_grouped_for_video(channels):
    features = []
    for n in range(0, 12):
        features.append([])
    for n in range(0, len(channels)):
        # if n == 1:
        #     # Solo un canale
        #     break
        start = 0
        end = 672
        for i in range(0, 12):
            signal_sample = np.array(channels[n][start:end])

            IMFs = EMD().emd(signal_sample, None, 1)

            # plot_IMFs(signal_sample, IMFs)

            #
            # print(IMFs)
            #
            # print(IMFs[1])

            ###### Per ogni canale, estrai le 3 features ######

            IMF = IMFs[0]
            D_t = get_first_difference_of_time_series(IMF)
            D_p = get_first_difference_of_phase(IMF)
            E_norm = get_normalized_energy(signal_sample, IMF)

            features[i].append(D_t)
            features[i].append(D_p)
            features[i].append(E_norm)

            start = end
            end += 672

    features = np.array(features)

    features_per_video = []

    for f in features:
        features_per_video.append(np.array(f))

    return features_per_video


def extract_features_separated(channels):
    samples = []
    features = []
    for n in range(0, 12):
        samples.append([])
        features.append([])
    for n in range(0, len(channels)):
        # if n == 1:
        #     # Solo un canale
        #     break
        start = 0
        end = 672
        for i in range(0, 12):
            signal_sample = np.array(channels[n][start:end])

            samples[i].append(signal_sample)

            IMFs = EMD().emd(signal_sample, None, 1)

            # plot_IMFs(signal_sample, IMFs)

            ###### Per ogni canale, estrai le 3 features ######

            IMF = IMFs[0]
            D_t = get_first_difference_of_time_series(IMF)
            D_p = get_first_difference_of_phase(IMF)
            E_norm = get_normalized_energy(signal_sample, IMF)

            feat = []

            feat.append(D_t)
            feat.append(D_p)
            feat.append(E_norm)

            features[i].append(np.array(feat))

            start = end
            end += 672

    # print(len(samples))

    samples = np.array(samples)

    features = np.array(features)

    set = []

    for f in features:
        set.append(np.array(f))

    return set


def plot_IMFs(signal_sample, IMFs):
    position = 1
    t = np.linspace(0, 1, len(signal_sample))

    # Plot results
    plt.subplot(2, 1, position)
    plt.plot(t, signal_sample, 'r')
    plt.title("Input signal")
    plt.xlabel("Time [s]")

    for n, imf in enumerate(IMFs):
        position += 1
        if n == 1:
            break
        plt.subplot(2, 1, position)
        plt.plot(t, imf, 'g')
        plt.title("IMF " + str(n + 1))
        plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig('simple_example')
    plt.show()


def print_to_file(participant, participant_predictions, correct_predictions, labels):
    # Stampa su file le statistiche del partecipante analizzato (o file f)
    with open('s' + participant + '_predictions.txt', 'w+') as file:
        text = str(participant_predictions) + "\n\n"
        for key, value in correct_predictions.items():
            text += str(key) + " correct predictions: " + str(value) + "/" + str(len(participant_predictions)) + "\n"

        text += "\n"

        for key, value in participant_predictions.items():
            text += "Video used as test set: " + str(key) + "\n"
            for k, v in value.items():
                index = 0
                if k == 'Arousal':
                    index = 1
                label = labels[key][index]
                predicted = "Low"
                if (v and label >= 5) or (not v and label < 5):
                    predicted = "High"

                space1 = "       "
                space2 = space1
                if v:
                    space1 += " "
                    space2 = "      "
                text += str(k) + " prediction: " + str(v) + space1 + "Label: " + str(label) + space2 + "Predicted: " + predicted + "\n"
            text += "\n"

        file.write(text)  # use `pickle.loads` to do the reverse


class Main:
    ###### Leggi il file dell'esperimento ed estrai i due array (labes e data) ######

    files = []

    # Files da analizzare e predire
    for n in range(8, 33):
        s = ''
        if n < 10:
            s += '0'
        s += str(n)
        files.append(s)

    print(files)

    # Per ogni partecipante classifica sia la valence che l'arousal per 40 volte, usando ogni volta un test set diverso
    for participant in files:
        filename = "E:\DatasetDEAP\DEAP\physiological recordings\data_preprocessed_python\s" + participant + ".dat"
        experiment = read_eeg_signal_from_file(filename)
        labels = experiment['labels']
        # print(labels)
        data = experiment['data']
        # print(data)

        # Features are extracted only once and stored per video into this array
        features_grouped_for_video = []

        for v in range(0, len(data)):
            channels = get_channels(v, data)

            features_grouped_for_video.append(extract_features_grouped_for_video(channels))


        # Un file corrisponde ad un partecipante
        # Per ogni partecipante faccio 40 predizioni, usando ogni volta un test set diverso
        # Perciò, per ogni partecipante creo una lista di predizioni, in cui verranno messe le predizioni
        # per ogni test set (sia per quanto riguarda la valence che l'arousal)
        # Un dizionario di dizionari (es di elemento: video 39 : ['Valence': True, 'Arousal': False])
        participant_predictions = {}

        # Conto le predizioni corrette
        correct_predictions = {'Valence': 0, 'Arousal': 0}

        leave_out = len(data) - 1   # inizialmente l'ultimo video è lasciato per il test set,
                                    # poi il penultimo e così via fino al primo

        # Ripeti 40 volte la classificazione, usando ogni volta un test set diverso
        while leave_out >= 0:
            X = []
            Y = []

            set = []

            # Inizializza il dizionario per valence e arousal relativo al video leave_out
            participant_predictions[leave_out] = {}

            # Per ogni video, estrai le features dagli 8 canali e inserisci il vettore ottenuto in X
            # Inserisci la label relativa al sample in Y
            for m in range(0, len(data)):
                if m == leave_out:
                    continue

                for f in features_grouped_for_video[m]:
                    X.append(f)

                frammenti = 12
                canali = 1
                max = frammenti*canali

            dtype = np.float64
            X = np.array(X, dtype=dtype)

            for i in range(0, 2):
                Y = []
                for m in range(0, len(data)):
                    if m == leave_out:
                        continue
                    for t in range(0, max):
                        label = 0
                        if labels[m][i] >= 5:
                            label = 1

                        Y.append(label)

                Y = np.array(Y, dtype=dtype)

                ###### Dai in input le features all'svm ######
                libsvm = svm.SVC()

                libsvm.fit(X, Y, None)

                T = []

                # Per ogni sample, estrai le features dagli 8 canali e inserisci il vettore ottenuto in X
                # Inserisci la label relativa al sample in Y
                # predict_channels = get_channels(leave_out, data)
                # T = extract_features_grouped_for_channel(T, predict_channels)

                for f in features_grouped_for_video[leave_out]:
                    T.append(f)

                dtype = np.float64

                # print(len(T))

                T = np.array(T, dtype=dtype)

                prediction = libsvm.predict(T)

                lab = 'Valence'
                if i == 1:
                    lab = 'Arousal'

                print("Video used as test set: " + str(leave_out))
                print(lab + " label: " + str(labels[leave_out][i]))
                print("Predicted values: " + str(prediction))

                correct_count = 0

                for pp in range(0, len(prediction)):
                    label = 0
                    if labels[leave_out][i] >= 5:
                        label = 1
                    if prediction[pp] == label:
                        correct_count += 1

                if correct_count > 6:
                    participant_predictions[leave_out][lab] = True
                    correct_predictions[lab] += 1
                else:
                    participant_predictions[leave_out][lab] = False


            leave_out -= 1

        # Stampa su file le statistiche del partecipante analizzato (o file f)
        print_to_file(participant, participant_predictions, correct_predictions, labels)