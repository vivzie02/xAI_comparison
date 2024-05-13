import os
import librosa
import numpy as np
import pandas as pd
from sympy.physics.quantum.identitysearch import scipy
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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


# region Model
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Assuming num_classes is defined
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
# endregion


def main():
    X, y = get_features('../data_loader/archive (15)/Data/genres_original/')
    genre_classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    y_indices = np.array([genre_classes.index(label) for label in y])

    num_classes = len(set(y))
    input_shape = (9, )
    X_train, X_test, y_train, y_test = train_test_split(X, y_indices, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    model = build_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    model.save('my_model.h5')
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
