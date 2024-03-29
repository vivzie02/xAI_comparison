import librosa
import numpy as np
import os
import torch


# takes the path to the audio files and returns a 4D tensor - a 3d image for each file (audio_files, width, height, channels)
def get_images(data_path):
    max_width = 1000
    max_height = 50
    images = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            print("Getting features for " + file)
            path = os.path.join(root, file)
            genre = os.path.basename(root)
            # sr is the sampling rate, y is the audio time series
            y, sr = librosa.load(path)

            # mfcc
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50, n_fft=255, hop_length=512, n_mels=20)
            mfcc = mfcc[:max_height, :max_width]
            # delta mfcc
            delta_mfcc = librosa.feature.delta(mfcc)
            # STFT (short time fourier)
            stft = np.abs(librosa.stft(y, n_fft=255))
            stft = stft[:max_height, :max_width]
            # chroma stft
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_stft = chroma_stft[:max_height, :max_width]

            mfcc_padded = np.pad(mfcc, ((0, max_height - mfcc.shape[0]), (0, max_width - mfcc.shape[1])), mode='constant')
            delta_mfcc_padded = np.pad(delta_mfcc, ((0, max_height - delta_mfcc.shape[0]), (0, max_width - delta_mfcc.shape[1])), mode='constant')
            stft_padded = np.pad(stft, ((0, max_height - stft.shape[0]), (0, max_width - stft.shape[1])), mode='constant')
            chroma_stft_padded = np.pad(chroma_stft, ((0, max_height - chroma_stft.shape[0]), (0, max_width - chroma_stft.shape[1])), mode='constant')

            images.append((mfcc_padded, delta_mfcc_padded, stft_padded, chroma_stft_padded))

    return np.array(images)


def main():
    data_path = './archive (15)/Data/genres_original/'

    images = get_images(data_path)
    tensor = torch.tensor(images)

    print("done")


if __name__ == "__main__":
    main()