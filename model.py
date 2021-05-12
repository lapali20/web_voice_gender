import librosa
import librosa.display
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import os
import entropy as ent

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import mode

class VoiceModel:
    def __init__(self, model_path='voicemodel.gz'):
        if os.path.exists(model_path):
            self.model = pickle.load(open(model_path, 'rb'))
        else:
            self.model = self.__train_model()
            pickle.dump(self.model, open(model_path, 'wb'))

    def __train_model(self):
        female_df = pd.read_csv('female.csv', index_col=0)
        male_df = pd.read_csv('male.csv', index_col=0)
        train_df = pd.concat([female_df, male_df], ignore_index=True)
        features = train_df.loc[:, train_df.columns != 'gender']
        labels = train_df['gender'].astype(np.int16)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=79)
        model = RandomForestClassifier().fit(X_train, y_train)
        return model

    def predict(self, features):
        y = self.model.predict(features)
        return y[0]

    def get_target_name(self, y):
        return ['Female', 'Male'][y]


def peak_freq(sound, sr):
    ft = sp.fft.fft(sound)
    magnitude = np.abs(ft)
    frequency = np.linspace(0, sr, len(magnitude))

    p_index = np.argmax(magnitude[:sr//2])
    peakfreq = frequency[p_index]
    return peakfreq / 1000


def med_freq(sound, rate):
    spec = np.fft.fft(sound)
    magnitude = np.abs(spec)
    frequency = np.linspace(0, rate, len(magnitude))
    power = np.sum(magnitude ** 2)

    mid = 0
    i = 0
    while mid < (power / 2):
        mid += magnitude[i] ** 2
        i += 1

    return frequency[i] / 1000


def create_time_frequency(sound, frame_size, hop_length, rate):
    s_scale = librosa.stft(sound, n_fft=frame_size, hop_length=hop_length, center=False)
    stft = []
    frequency = np.arange(0, 1 + frame_size/2) * rate / frame_size

    for i in range(s_scale.shape[1]):
        seg = s_scale.transpose()[i]
        magnitude = np.abs(seg)

        index = np.argmax(magnitude)
        stft.append(frequency[index] / 1000)

    return np.asarray(stft)


def create_features(sound, sr):
    X = pd.DataFrame()
    segment_id = 0
    FRAME_SIZE = 256
    HOP_LENGTH = 128

    tf = create_time_frequency(sound, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, rate=sr)
    centroid = librosa.feature.spectral_centroid(sound, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH).transpose()

    X.loc[segment_id, 'centroid'] = np.min(centroid) / 1000
    X.loc[segment_id, 'meanfreq'] = med_freq(sound, rate=sr)
    X.loc[segment_id, 'sd'] = np.std(tf)
    X.loc[segment_id, 'kurt'] = kurtosis(tf)
    X.loc[segment_id, 'skew'] = skew(tf)
    X.loc[segment_id, 'mode'] = mode(tf).mode[0]
    X.loc[segment_id, 'peakfreq'] = peak_freq(sound, sr)
    X.loc[segment_id, 'Q25'] = q25 = np.quantile(tf, 0.25)
    X.loc[segment_id, 'Q75'] = q75 = np.quantile(tf, 0.75)
    X.loc[segment_id, 'IQR'] = q75 - q25
    X.loc[segment_id, 'sp.ent'] = ent.spectral_entropy(sound, sf=sr)
    X.loc[segment_id, 'sfm'] = np.std(librosa.feature.spectral_flatness(sound, n_fft=FRAME_SIZE, hop_length=128))
    X.loc[segment_id, 'mindom'] = np.min(tf)

    return X
