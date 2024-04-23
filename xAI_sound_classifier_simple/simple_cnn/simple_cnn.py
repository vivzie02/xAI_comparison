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
from PIL import Image as im
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
    path = f'./testdata/images/{name}/'
    file = os.path.splitext(file)[0] + '.png'
    plt.figure(figsize=(25, 5))
    librosa.display.specshow(feature, x_axis='time')
    plt.title(name)
    plt.colorbar()
    plt.savefig(path + file)
    plt.close()


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


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()


def main():
    data_path = '../data_loader/archive (15)/Data/genres_original/'

    print(torch.cuda.is_available())

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
            inputs, labels = inputs.to(device), labels.to(device)
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        loss_total_training.append(np.array(epoch_loss).mean())

        count = 0
        correct = 0
        y_true = []
        all_predictions = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_pred = model(inputs)
            _, predicted_classes = torch.max(y_pred, 1)  # Get the index of the maximum logit as the predicted class
            all_predictions.extend(predicted_classes.tolist())
            y_true.extend(labels.tolist())
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
    torch.save(model.state_dict(), "CNNModel.pth")

    plot_confusion_matrix(y_true, all_predictions, label_encoder.classes_)


if __name__ == "__main__":
    main()
