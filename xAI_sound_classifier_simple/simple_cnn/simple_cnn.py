import librosa
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# region Preprocessing

def pad_and_normalize(feature, max_height, max_width, name):
    # normalize over each individual image feature
    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
    '''
    plt.figure(figsize=(25, 5))
    librosa.display.specshow(feature, x_axis='time')
    plt.title(name)
    plt.colorbar()
    '''
    return np.pad(feature, ((0, max_height - feature.shape[0]), (0, max_width - feature.shape[1])), mode='constant')


# takes the path to audio files and returns a 4D tensor-a 3d image for each file (audio_files, width, height, channels)
def get_features(data_path):
    max_width = 100
    max_height = 20
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
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=max_height, n_fft=255, hop_length=512, n_mels=50)
            mfcc = mfcc[:max_height, :max_width]
            # delta mfcc
            delta_mfcc = librosa.feature.delta(mfcc)
            # STFT (short time fourier)
            stft = np.abs(librosa.stft(y, n_fft=255))
            stft = stft[:max_height, :max_width]
            # chroma stft
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_stft = chroma_stft[:max_height, :max_width]

            mfcc = pad_and_normalize(mfcc, max_height, max_width, "MFCC")
            delta_mfcc = pad_and_normalize(delta_mfcc, max_height, max_width, "Delta MFCC")
            stft = pad_and_normalize(stft, max_height, max_width, "STFT")
            chroma_stft = pad_and_normalize(chroma_stft, max_height, max_width, "Chromogram STFT")

            images.append((mfcc, delta_mfcc, stft, chroma_stft))
            genres.append(genre)

    return np.array(images), genres
# endregion


# region CNN

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc3 = nn.Linear(16000, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input x as (4x20x100)
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # second layer
        x = self.act2(self.conv2(x))
        # (32x20x100) to (32x10x50)
        x = self.pool2(x)
        # third layer (32x10x50) to 16000
        x = self.flatten(x)
        # fourth layer 16000 to 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # final layer 512 to 10
        x = self.fc4(x)
        return x

# endregion


def main():
    data_path = '../data_loader/archive (15)/Data/genres_original/'

    # get audio feature images
    images, genres = get_features(data_path)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # train the network
    label_encoder = LabelEncoder()

    X_train, X_test, y_train, y_test = train_test_split(images, genres, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(label_encoder.fit_transform(y_train))
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(label_encoder.transform(y_test))

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 32
    n_epochs = 10

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    model = Network()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = 0
        count = 0
        for inputs, labels in test_loader:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
        acc /= count
        print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

    torch.save(model.state_dict(), "CNNModel.pth")


if __name__ == "__main__":
    main()
