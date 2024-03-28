import librosa
import numpy as np
import os
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler


class AudioFeatures:
    def __init__(self):
        # general info
        self.file = None
        self.sampling_rate = None
        self.time_series = None
        self.genre = None

        self.energy = None
        self.zero_crossings = None
        self.tempo = None
        self.mfcc = None
        self.mfcc_delta = None

        # statistical data
        self.mean = None
        self.std = None
        self.maxv = None
        self.minv = None
        self.median = None
        self.skew = None
        self.kurt = None
        self.q1 = None
        self.q3 = None
        self.iqr = None


def get_wave_data(data_path):
    raw_data = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            print("Getting features for " + file)
            features = AudioFeatures()

            path = os.path.join(root, file)
            genre = os.path.basename(root)
            # sr is the sampling rate, y is the audio time series
            y, sr = librosa.load(path)

            features.file = file
            features.genre = genre
            features.sampling_rate = sr
            features.time_series = y

            features.energy = np.sum(y ** 2)
            features.zero_crossings = sum(librosa.zero_crossings(y, pad=False))
            features.tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # MFCC features (mean of time windows)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
            features.mfcc = mfcc
            features.mfcc_delta = librosa.feature.delta(mfcc)

            raw_data.append(features)

    return raw_data


def get_statistical_features(audio_data):
    for file_data in audio_data:
        print("Get stats for", file_data)

        # get the amplitudes via fft (abs)
        magnitudes = np.abs(np.fft.fft(file_data.time_series))

        file_data.mean = np.mean(magnitudes)
        file_data.std = np.std(magnitudes)
        file_data.maxv = np.amax(magnitudes)
        file_data.minv = np.amin(magnitudes)
        file_data.median = np.median(magnitudes)
        file_data.skew = scipy.stats.skew(magnitudes)
        file_data.kurt = scipy.stats.kurtosis(magnitudes)
        file_data.q1 = np.quantile(magnitudes, 0.25)
        file_data.q3 = np.quantile(magnitudes, 0.75)
        file_data.iqr = scipy.stats.iqr(magnitudes)

    return audio_data


def get_features(data_path):
    # extract audio time series from the files (raw wave data)
    audio_data = get_wave_data(data_path)

    # extract statistical features from wave data
    audio_data = get_statistical_features(audio_data)

    df = pd.DataFrame.from_records(vars(data) for data in audio_data)
    return df


def preprocess_data(df):
    # normalize
    scaler = MinMaxScaler()
    non_normalized_cols = ['sampling_rate', 'file', 'genre', 'mfcc']
    # regular normalization
    for column in df.columns:
        if column in non_normalized_cols:
            continue
        df[column] = scaler.fit_transform(df[column])

    return df


def main():
    data_path = './archive (15)/Data/genres_original/'

    # get MFCC, energy, ...
    df = get_features(data_path)

    # data preprocessing
    df = preprocess_data(df)


if __name__ == "__main__":
    main()