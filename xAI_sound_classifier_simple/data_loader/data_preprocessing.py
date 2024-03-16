import torch
import torchaudio
import librosa
import numpy as np
import os
import pandas as pd


def build_network():
    pass


def main():
    data_path = './archive (15)/Data/genres_original/'
    columns = ['name', 'time_series', 'sampling_rate', 'genre']
    data = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            path = os.path.join(root, file)
            genre = os.path.basename(root)
            # sr is the sampling rate, y is the audio time series
            y, sr = librosa.load(path)
            data.append([file, y, sr, genre])

    df = pd.DataFrame(data, columns=columns)
    print(df)


if __name__ == "__main__":
    main()