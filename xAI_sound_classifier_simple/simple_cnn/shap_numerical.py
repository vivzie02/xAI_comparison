import glob

import cv2
import librosa
import numpy as np
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from tensorflow.keras import layers, models
import tensorflow as tf


# region Feature-Extraction
def pad(feature, max_height, max_width, name):
    '''
    plt.figure(figsize=(25, 5))
    librosa.display.specshow(feature, x_axis='time')
    plt.title(name)
    plt.colorbar()
    '''
    return np.pad(feature, ((0, max_height - feature.shape[0]), (0, max_width - feature.shape[1])), mode='constant')


def get_features(data_path):
    max_width = 1000
    max_height = 40
    images = []
    genres = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            print("Getting features for " + file)
            path = os.path.join(root, file)
            genre = os.path.basename(root)
            # sr is the sampling rate, y is the audio time series
            y, sr = librosa.load(path)

            # Extract features
            features = {}

            # Time-domain features
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['energy'] = np.sum(librosa.feature.rms(y=y))
            features['rms_energy'] = np.mean(librosa.feature.rms(y=y))

            # Frequency-domain features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)

            spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidths)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidths)

            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            features['spectral_flatness_std'] = np.std(spectral_flatness)

            # Add more features as needed
            images.append(features)
            genres.append(genre)

    return pd.DataFrame(images), genres

# endregion


# region Feature-Extraction
def pad(feature, max_height, max_width, name):
    '''
    plt.figure(figsize=(25, 5))
    librosa.display.specshow(feature, x_axis='time')
    plt.title(name)
    plt.colorbar()
    '''
    return np.pad(feature, ((0, max_height - feature.shape[0]), (0, max_width - feature.shape[1])), mode='constant')


def get_features(data_path):
    max_width = 1000
    max_height = 40
    images = []
    genres = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            print("Getting features for " + file)
            path = os.path.join(root, file)
            genre = os.path.basename(root)
            # sr is the sampling rate, y is the audio time series
            y, sr = librosa.load(path)

            # Extract features
            features = {}

            # Time-domain features
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['energy'] = np.sum(librosa.feature.rms(y=y))
            features['rms_energy'] = np.mean(librosa.feature.rms(y=y))

            # Frequency-domain features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)

            spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidths)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidths)

            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            features['spectral_flatness_std'] = np.std(spectral_flatness)

            # Add more features as needed
            images.append(features)
            genres.append(genre)

    return pd.DataFrame(images), genres

# endregion


model = tf.keras.models.load_model('my_model.h5')


def main():
    X, y = get_features('../data_loader/archive (15)/Data/genres_original/')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    explainer = shap.DeepExplainer(model, np.array(X_train))
    shap_values = explainer.shap_values(np.array(X_test))

    shap.summary_plot(shap_values, np.array(X_test), feature_names=X_test.columns, class_names=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
    # Bar Plot
    shap.summary_plot(shap_values, np.array(X_test), plot_type='bar', feature_names=X_test.columns, class_names=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])


if __name__ == "__main__":
    main()
