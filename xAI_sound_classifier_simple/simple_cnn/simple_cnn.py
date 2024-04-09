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

def pad(feature, max_height, max_width, name):
    '''
    plt.figure(figsize=(25, 5))
    librosa.display.specshow(feature, x_axis='time')
    plt.title(name)
    plt.colorbar()
    '''
    return np.pad(feature, ((0, max_height - feature.shape[0]), (0, max_width - feature.shape[1])), mode='constant')


# takes the path to audio files and returns a 4D tensor-a 3d image for each file (audio_files, width, height, channels)
def get_features(data_path):
    max_width = 1000
    max_height = 50
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

            mfcc = pad(mfcc, max_height, max_width, "MFCC")
            delta_mfcc = pad(delta_mfcc, max_height, max_width, "Delta MFCC")
            stft = pad(stft, max_height, max_width, "STFT")
            chroma_stft = pad(chroma_stft, max_height, max_width, "Chromogram STFT")

            images.append((mfcc, delta_mfcc, stft, chroma_stft))
            genres.append(genre)

    return np.array(images), genres
# endregion


# region CNN

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512 * 3 * 62, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# endregion


def main():
    data_path = '../data_loader/archive (15)/Data/genres_original/'

    # get audio feature images
    images, genres = get_features(data_path)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # train the network
    label_encoder = LabelEncoder()

    # split into test/train and normalize
    X_train, X_test, y_train, y_test = train_test_split(images, genres, test_size=0.2, random_state=42)

    # X_train = np.array((X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train)))
    # X_train = torch.tensor(X_train / np.std(X_train))
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(label_encoder.fit_transform(y_train))

    # X_test = np.array((X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test)))
    # X_test = torch.tensor(X_test / np.std(X_test))
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(label_encoder.transform(y_test))

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32
    n_epochs = 40

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    model = Network().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_total_training = []
    accuracy_scores = []

    for epoch in range(n_epochs):
        epoch_loss = []

        for inputs, labels in train_loader:
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        loss_total_training.append(np.array(epoch_loss).mean())

        count = 0
        correct = 0
        for inputs, labels in test_loader:
            y_pred = model(inputs)
            _, predicted_classes = torch.max(y_pred, 1)  # Get the index of the maximum logit as the predicted class
            correct += (predicted_classes == labels).sum().item()
            count += len(labels)
        acc = correct / count * 100
        accuracy_scores.append(acc)
        print("Epoch %d: model accuracy %.2f%%" % (epoch, acc))

    plt.plot(loss_total_training, label='train loss')
    plt.legend()
    plt.show()

    plt.plot(accuracy_scores, label='Accuracy')
    plt.legend()
    plt.show()
    # save model


if __name__ == "__main__":
    main()
