import glob

import cv2
import librosa
import numpy as np
import os
import torch
import torch.nn as nn
from matplotlib import image as mpimg
from numpy.random import randint
from random import choice
import matplotlib.pyplot as plt
from torch.utils import data
from torch.utils.data import TensorDataset


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
    filepaths = []
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

            chroma_stft = chroma_stft[:, :max_width]
            chroma_stft = pad(chroma_stft, max_height, max_width, "Chromogram STFT")
            # chroma_stft = np.expand_dims(chroma_stft, axis=0)

            filepath = save_feature_image(mel_spectrogram_padded, "Mel_Spectrogram", file)
            mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=0)
            filepaths.append(filepath)

            images.append(mel_spectrogram_padded)
            genres.append(genre)

    return np.array(images), filepaths


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


# region Grad-CAM
def make_gradcam_heatmap(img_tensor, model, last_conv_layer_name, pred_index):
    # Create a sub-model that outputs the feature maps and final prediction
    class SubModel(nn.Module):
        def __init__(self, base_model, last_conv_layer_name):
            super(SubModel, self).__init__()
            self.base_model = base_model
            self.last_conv_layer_name = last_conv_layer_name

        def forward(self, x):
            # Get the output of the specified layer
            submodel = nn.Sequential(*list(model.children())[:13])
            conv_output = submodel(x)

            # Get the final output of the model
            final_output = self.base_model(x)

            return conv_output, final_output

    grad_model = SubModel(model, last_conv_layer_name)

    # Set the model to evaluation mode
    grad_model.eval()

    # Use autograd to record gradients
    img_tensor.requires_grad_(True)
    conv_output, preds = grad_model(img_tensor)

    # Gather the class score corresponding to the predicted class
    class_score = preds[:, pred_index]

    # Backpropagate
    grad_model.zero_grad()
    class_score.backward()

    # Get the gradients from the last convolutional layer
    gradients = grad_model.base_model.conv10.weight.grad

    # Pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get the activations of the last convolutional layer (activations)
    last_conv_layer_output = conv_output.detach()

    for i in range(256):
        last_conv_layer_output[:, i, :, :] *= pooled_gradients[i]

    # get heatmap
    heatmap = torch.mean(last_conv_layer_output, dim=1).squeeze()

    # Normalise the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    # plt.matshow(heatmap.squeeze())
    # plt.show()

    return heatmap
# endregion


def plot_gradcam_heatmaps(img, model, genres, outputs, heatmaps):
    outputs = torch.softmax(outputs, dim=0)
    # cv2 uses BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))

    for k in range(len(genre_classes)):
        # try heatmap by changing visibility of the colors
        heatmap = heatmaps[k]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        superimposed_img = img

        # can't see anything, display side by side instead
        # superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 1, 0)

        plt.subplot(5, 4, k*2 + 1)
        plt.imshow(heatmap, cmap='hot')
        plt.axis('off')
        plt.title(f"Grad-CAM Heatmap for {genre_classes[k]}", fontsize=8)

        plt.subplot(5, 4, k*2 + 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted {genre_classes[k]} with a score of {outputs[k]}", fontsize=8)

        plt.tight_layout()

    plt.show()


testdata_path = './testdata/audio/'
genre_classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

images, filepaths = get_features(testdata_path)

images = torch.tensor(images)

model = Network()
model.load_state_dict(torch.load("CNNModel.pth"))
model.eval()

output = model(images)

predicted_label = torch.argmax(output, dim=1)

# load images into a data loader
image_loader = data.DataLoader(dataset=images, shuffle=False, batch_size=1)

for i, label in enumerate(predicted_label):
    image = next(iter(image_loader))
    heatmaps = []

    for k in range(len(genre_classes)):
        heatmaps.append(make_gradcam_heatmap(image, model, 'conv10', pred_index=k).detach().numpy())

    img = cv2.imread(filepaths[i])
    cv2.imshow("Test", img)

    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    plot_gradcam_heatmaps(img, model, genre_classes, output[i], heatmaps)

    # try heatmap by changing visibility of the colors
    '''
    superimposed_img = img
    for i in range(3):
        superimposed_img[:, :, i] = np.uint8(np.minimum(img[:, :, i] * heatmap, 255))

    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 1, 0)

    cv2.imshow('Heatmap', heatmap)
    cv2.waitKey(0)

    cv2.imshow('Image with Heatmap', superimposed_img)
    cv2.waitKey(0)
    
    '''

    print("Predicted labels:", genre_classes[label.item()])
