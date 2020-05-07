
import cv2
import os
import math
import random
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import argparse

#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, ZeroPadding3D, BatchNormalization, Activation, Attention
from tensorflow.keras.layers import LSTM, TimeDistributed,Conv2D, MaxPooling3D, Conv3D,MaxPooling2D
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from skimage.io import imread, imread_collection, concatenate_images, ImageCollection
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet50
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

file_id = '1pb7fTGKh39h4cUBpkX1Q2qK9K8zWJ6iD'
destination = 'model.hdf5'
download_file_from_google_drive(file_id, destination)
fr_vid = None
frame_count = None

def frame_extractor(video):
    if not os.path.exists('Test_set'):
        os.mkdir('Test_set')
    frames_path = 'Test_set/'
    video_name, video_ext = os.path.splitext(video)
    path = frames_path + video_name + '_frames'
    if not os.path.exists(path):
        os.mkdir(path)
    count = 0
    cap = cv2.VideoCapture(video)  # capturing the video from the given path
    x = 1
    frameRate = cap.get(5)  # frame rate
    fr_vid = frameRate
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        filename = path + "/frame%d.jpg" % count
        count += 1
        cv2.imwrite(filename, frame)
    cap.release()
    frame_count = count
    print("Done!")


parser = argparse.ArgumentParser()
parser.add_argument('--video_name')
args = parser.parse_args()
print('Extracting Frames for the given Video!')
frame_extractor(args.video_name)

# JSON EXTRACTION


test_path = 'Test_set/'
# label_p = 'data.json'
model_path = 'model.hdf5'
json_out_path = 'timeLabels.json'

df2 = None # pd.read_json(label_p)


class DataGenerator(Sequence):
    def __init__(self, path, video_list, label_df, to_fit=True, batch_size=5, dim=(290, 480), n_channels=3,
                 n_classes=30, shuffle=True):
        """Parameters
    : video_list: list of all videos
    : labels: list of image labels (file names)
    : path: path to location of videos
    : label_path: path to label json file
    : to_fit: True to return X(batch video frames) and y(labels), False to return X only
    : batch_size: batch size at each iteration
    : dim: tuple indicating image dimension
    : n_channels: number of image channels
    : n_classes: number of output classes also same as time dimension
    : shuffle: True to shuffle label indexes after every epoch
    """
        self.path = path
        self.video_list = video_list
        self.label_df = label_df
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        if self.batch_size % 2 == 0:
            return int(np.floor(len(self.video_list) / self.batch_size))
        else:
            return int(np.ceil(len(self.video_list) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        video_list_temp = [self.video_list[k] for k in indexes]

        # Generate data
        X = self._generate_X(video_list_temp)

        if self.to_fit:
            y = self._generate_y(video_list_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.video_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, video_list_temp):
        X = np.empty((self.batch_size, self.n_classes, *self.dim, self.n_channels))
        # Generate data
        for i, VID in enumerate(video_list_temp):
            # Store sample
            X[i, :] = self._load_frames(VID)
        return X

    def _load_frames(self, vid):
        video_path = self.path + vid
        frame_paths = os.listdir(video_path)
        frame_input1 = [imread(video_path + '/' + fr_path) / 255 for fr_path in frame_paths]
        if (len(frame_input1) < self.n_classes):
            diff = math.ceil((self.n_classes - len(frame_input1)) / 2)
            frame_input = [None] * self.n_classes
            # Repeating First element
            frame_input[0:diff] = [frame_input1[0]] * diff
            # Centering frames
            frame_input[diff:diff + len(frame_input1)] = frame_input1
            # Repeating last frames
            frame_input[diff + len(frame_input1):] = [frame_input1[-1]] * (self.n_classes - len(frame_input1) - diff)
        elif (len(frame_input1) > self.n_classes):
            if self.label_df == None:
                frame_input = frame_input1[0:30]
            else:
                idx = self.get_labels(vid, need_idx=1)
                frame_input = frame_input1[idx[0]:idx[1]]
        else:
            frame_input = frame_input1
        frame_input = np.array(frame_input)
        if np.sum(np.isnan(frame_input)) > 0:
            print(np.sum(np.isnan(frame_input)))
        return frame_input

    def _generate_y(self, video_list_temp):
        y = np.empty([self.batch_size, self.n_classes, 2], dtype=float)

        # Generate data
        for i, VID in enumerate(video_list_temp):
            # Store sample
            y[i, :] = self.get_labels(VID)

        return y

    def get_labels(self, label_file, need_idx=False):

        frame_paths = os.listdir(self.path + label_file)
        n_f = len(frame_paths)
        df = self.label_df
        labels = df.loc[label_file].values
        where_are_nans = np.isnan(labels)
        labels[where_are_nans] = 0
        labels = np.squeeze(labels)
        n_l = self.n_classes
        if n_l > n_f:
            diff = math.ceil((n_l - n_f) / 2)
            output = [None] * n_l
            # Repeating First element
            output[0:diff] = [labels[0]] * diff
            # Centering frames
            output[diff:diff + n_f] = labels[0:n_f]
            # Repeating last frames
            output[diff + n_f:n_l] = [labels[n_f]] * (n_l - n_f - diff)
        elif n_l < n_f:
            if sum(labels) > 0:
                where_are_ones = np.isin(labels, 1)
                idxs = np.nonzero(where_are_ones)
                idxs = np.squeeze(idxs)
                l = labels[where_are_ones]
                diff = math.ceil((n_l - len(l)) / 2)
                d1 = idxs[0] - diff
                d2 = idxs[-1] + diff
                if d1 < 0:
                    idx = [0, n_l]
                    output = labels[idx[0]:idx[1]]
                elif d2 > n_f:
                    idx = [n_f - n_l, n_f]
                    output = labels[idx[0]:idx[1]]
                else:
                    diff21 = math.floor((n_l - idxs[-1] + idxs[0]) / 2)
                    diff22 = math.ceil((n_l - idxs[-1] + idxs[0]) / 2)
                    diff23 = diff22 - diff21
                    idx = [idxs[0] - diff21, idxs[-1] + diff21 + diff23]
                    output = labels[idx[0]:idx[1]]
            else:
                diff = math.ceil((n_f - n_l) / 2)
                idx = [diff, diff + n_l]
                output = labels[diff:diff + n_l]
            if need_idx == 1:
                return idx
        else:
            output = labels[0:n_l]
        output = np.expand_dims(output, axis=1)
        output1 = 1 - output
        output2 = np.concatenate([output, output1], axis=1)
        if np.sum(np.isnan(output)) > 0:
            print(np.sum(np.isnan(output)))
        return (output2)

model2 = load_model(model_path)

test_list = os.listdir(test_path)
predict_frame_gen = DataGenerator(test_path, test_list, None, batch_size=1, shuffle=False, to_fit=False)
y_pred = model2.predict_generator(predict_frame_gen)
print(y_pred)


def get_times(y1, y2, vid, fps):
    frame_paths = os.listdir(test_path + vid)
    n_f = len(frame_paths)
    df = df2
    labels = y1
    where_are_nans = np.isnan(labels)
    labels[where_are_nans] = 0
    labels = np.squeeze(labels)
    n_l = len(y2)
    t1 = np.arange(len(y1)) / fps
    if n_l > n_f:
        diff = math.ceil((n_l - n_f) / 2)
        output = [None] * n_l
        t = [None] * n_l
        # Repeating First element
        output[0:diff] = [labels[0]] * diff
        t[0:diff] = [t1[0]] * diff
        # Centering frames
        output[diff:diff + n_f] = labels[0:n_f]
        t[diff:diff + n_f] = t1[0:n_f]
        # Repeating last frames
        output[diff + n_f:] = [labels[n_f]] * (n_l - n_f - diff)
        t[diff + n_f:] = [t1[n_f]] * (n_l - n_f - diff)
    elif n_l < n_f:
        if sum(labels) > 0:
            where_are_ones = np.isin(labels, 1)
            idxs = np.nonzero(where_are_ones)
            idxs = np.squeeze(idxs)
            l = labels[where_are_ones]
            diff = math.ceil((n_l - len(l)) / 2)
            d1 = idxs[0] - diff
            d2 = idxs[-1] + diff
            if d1 < 0:
                idx = [0, n_l]
                output = labels[idx[0]:idx[1]]
                t = t1[idx[0]:idx[1]]
            elif d2 > n_f:
                # d1 = d1-d2+n_l
                idx = [n_f - n_l, n_f]
                output = labels[idx[0]:idx[1]]
                t = t1[idx[0]:idx[1]]
            else:
                diff21 = math.floor((n_l - idxs[-1] + idxs[0]) / 2)
                diff22 = math.ceil((n_l - idxs[-1] + idxs[0]) / 2)
                diff23 = diff22 - diff21
                idx = [idxs[0] - diff21, idxs[-1] + diff21 + diff23]
                # print('idx',idx)
                output = labels[idx[0]:idx[1]]
                t = t1[idx[0]:idx[1]]
            # print("D2-D1",d2-d1)
            # print(labels[idx[0]:idx[1]])
        else:
            diff = math.ceil((n_f - n_l) / 2)
            idx = [diff, diff + n_l]
            output = labels[idx[0]:idx[1]]
            t = t1[idx[0]:idx[1]]
    else:
        output = labels[0:n_l]
        t = t1[0:n_l]
    output = np.expand_dims(output, axis=1)
    output1 = 1 - output
    output2 = np.concatenate([output, output1], axis=1)
    t = np.squeeze(np.array(t))
    return t, output


tl2 = []
y1_adj2 = list()
y_diff = list()
count = 0
for i, vid in enumerate(test_list):
    if df2==None:
        y2 = y_pred[i]
        y2x = y2[:,0]
        tl = []
        for j in range(len(y2)):
            tl.append([j*frame_count/fr_vid, y2x[j]])
        tl2.append(tl)
        plt.figure()
        plt.plot(y2x)
        plt.ylim(-0.02, 1.2)
        name = vid + ' Label plot'
        plt.title(name)
        plt.ylabel('Label')
        plt.xlabel('Time')
        plt.legend(['Actual'], loc='upper right')
        # plt.show()
        plt.savefig('timeLabel.jpg')
        plt.close
    else:
        y1 = df2.loc[vid].values
        y2 = y_pred[i]
        t2, y1_adj = get_times(y1, y2, vid, 10)
        y2x = y2[:, 0]
        y1_adjx = y1_adj[:, 0]
        plt.figure()
        plt.plot(t2, y2x)
        plt.plot(t2, y1_adjx)
        plt.ylim(-0.02, 1.2)
        name = vid + ' Label plot'
        plt.title(name)
        plt.ylabel('Label')
        plt.xlabel('Time')
        plt.legend(['Predicted', 'Actual'], loc='upper right')
        # plt.show()
        plt.savefig(timelabel_fig_path + vid + '.jpg')
        plt.close
        tl = []
        for j in range(len(y2)):
            tl.append([t2[j], y2x[j]])
        tl2.append(tl)
        y3 = np.zeros(y2x.shape)
        y3_adj = np.zeros(y1_adjx.shape)
        y3[y2x >= 0.5] = 1
        y3_adj[y1_adjx >= 0.5] = 1
        y_diff += [np.sum(np.abs(y3 - y3_adj)) / len(y3)]
        y1_adj2 += [y3_adj]
        if list(y3) == list(y3_adj):
            count += 1
d = {}
for i,vid in enumerate(test_list):
  d[vid] = tl2[i]
tL_df = pd.DataFrame(d)
tL_df.to_json(json_out_path)
