import glob

import cv2
import librosa
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
from matplotlib import image as mpimg
from numpy.random import randint
from random import choice
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import shap


# region Preprocessing
def pad(feature, max_height, max_width, name):
    '''
    plt.figure(figsize=(25, 5))
    librosa.display.specshow(feature, x_axis='time')
    plt.title(name)
    plt.colorbar()
    '''
    return np.pad(feature, ((0, max_height - feature.shape[0]), (0, max_width - feature.shape[1])), mode='constant')


def save_feature_image(feature, name, file):
    path = f'./testdata/temp/'
    file = os.path.splitext(file)[0] + '.png'
    filepath = path + file

    plt.figure(frameon=False)
    plt.axis('off')
    librosa.display.specshow(feature)
    plt.savefig(path + file, bbox_inches='tight', pad_inches=0)
    plt.close()

    return filepath


# takes the path to audio files and returns a 4D tensor-a 3d image for each file (audio_files, width, height, channels)
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

            # mfcc
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=max_height, n_fft=512, hop_length=160, n_mels=40)
            mfcc = mfcc[:, :max_width]
            mfcc = pad(mfcc, max_height, max_width, "MFCC")
            # mfcc = np.expand_dims(mfcc, axis=0)
            # delta mfcc
            delta_mfcc = librosa.feature.delta(mfcc)
            # chroma stft
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            '''
            # STFT (short time fourier)
            stft = np.abs(librosa.stft(y, n_fft=255))
            stft = stft[:max_height, :max_width]

            delta_mfcc = pad(delta_mfcc, max_height, max_width, "Delta MFCC")
            stft = pad(stft, max_height, max_width, "STFT")
            '''
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=255, n_mels=max_height)
            # Convert to dB scale for better visualization
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_spectrogram = mel_spectrogram[:, :max_width]
            mel_spectrogram_padded = pad(mel_spectrogram, max_height, max_width, "Mel Spectrogram")
            mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=0)

            chroma_stft = chroma_stft[:, :max_width]
            chroma_stft = pad(chroma_stft, max_height, max_width, "Chromogram STFT")
            # chroma_stft = np.expand_dims(chroma_stft, axis=0)

            images.append(mel_spectrogram_padded)
            genres.append(genre)

    return np.array(images), genres


# endregion


# region CNN
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv11 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn11 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv13 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn13 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv7 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv12 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn12 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.pool8 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv9 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.pool10 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256 * 1 * 31, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn11(self.conv11(x)))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn13(self.conv13(x)))
        x = self.pool6(torch.relu(self.bn6(self.conv6(x))))
        x = torch.relu(self.bn7(self.conv7(x)))
        x = torch.relu(self.bn12(self.conv12(x)))
        x = self.pool8(torch.relu(self.bn8(self.conv8(x))))
        x = torch.relu(self.bn9(self.conv9(x)))
        x = self.pool10(torch.relu(self.bn10(self.conv10(x))))

        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
# endregion


device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = Network().to(device)
model.load_state_dict(torch.load("CNNModel.pth"))
model.eval()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 1 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 1 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 1 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 1 else x.permute(1, 2, 0)
    return x


testdata_path = './testdata/audio/'
data_path = '../data_loader/archive (15)/Data/genres_original/'
genre_classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

test_images, test_genres = get_features(testdata_path)
images, genres = get_features(data_path)

test_images = torch.tensor(test_images)
X_train, X_test, y_train, y_test = train_test_split(images, genres, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train)


mean = torch.mean(X_train, dim=(2, 3))[0]
std = torch.std(X_train, dim=(2, 3))[0]

transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
    torchvision.transforms.Normalize(mean=mean, std=std),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

inv_transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist(),
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)


def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img))
    img = img.to(device)
    output = model(img)
    return output


def main():
    # test
    Xtr = transform(torch.Tensor(X_test))
    out = predict(Xtr[1:3])
    classes = torch.argmax(out, axis=1).cpu().numpy()
    print(f"Classes: {classes}: {np.array(genre_classes)[classes]}")

    # test
    masker_blur = shap.maskers.Image("blur(128, 128)", Xtr[0].shape)

    explainer = shap.Explainer(predict, masker_blur, output_names=genre_classes)
    shap_values = explainer(
        Xtr[:1],
        max_evals=100,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:10]
    )

    outputs = predict(Xtr[3:20])
    classes = torch.argmax(outputs, axis=1).cpu().numpy()
    print(f"Classes: {classes}: {np.array(genre_classes)[classes]}")
    for index, output in enumerate(outputs[3:20]):
        print(f"actual output {y_train[index + 3]}")


if __name__ == "__main__":
    main()
