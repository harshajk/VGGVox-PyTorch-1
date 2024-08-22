import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import signal_utils as sig
from scipy.io import wavfile
from vggm import VGGM
import argparse

LR = 0.01
B_SIZE = 100
N_EPOCHS = 150
N_CLASSES = 1251
LOCAL_DATA_DIR = "data/"
MODEL_DIR = "models/"

class AudioDataset(tf.keras.utils.Sequence):
    def __init__(self, csv_file, data_dir, croplen=48320, is_train=True, batch_size=B_SIZE):
        if isinstance(csv_file, str):
            csv_file = pd.read_csv(csv_file)
        assert isinstance(csv_file, pd.DataFrame), "Invalid csv path or dataframe"
        self.X = csv_file['Path'].values
        self.y = (csv_file['Label'].values - 10001).astype(int)
        self.data_dir = data_dir
        self.is_train = is_train
        self.croplen = croplen
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        audio_batch = []
        for file_path in batch_x:
            sr, audio = wavfile.read(os.path.join(self.data_dir, file_path))
            if self.is_train:
                start = np.random.randint(0, audio.shape[0] - self.croplen + 1)
                audio = audio[start:start + self.croplen]
            audio = sig.preprocess(audio).astype(np.float32)
            audio = np.expand_dims(audio, axis=-1)
            audio_batch.append(audio)
        
        return np.array(audio_batch), np.array(batch_y)

def accuracy(y_true, y_pred, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = tf.shape(y_true)[0]
    
    _, top_k_indices = tf.math.top_k(y_pred, k=maxk)
    correct = tf.equal(tf.expand_dims(y_true, -1), tf.cast(top_k_indices, y_true.dtype))
    
    results = []
    for k in topk:
        correct_k = tf.reduce_sum(tf.cast(tf.reduce_any(correct[:, :k], axis=-1), tf.float32))
        results.append(correct_k * (100.0 / tf.cast(batch_size, tf.float32)))
    
    return results

def test(model, datasets):
    corr1 = 0
    corr5 = 0
    counter = 0
    top1 = 0
    top5 = 0
    for dataset in datasets:
        sub_counter = 0
        sub_top1 = 0
        sub_top5 = 0
        for audio, labels in dataset:
            outputs = model(audio, training=False)
            corr1, corr5 = accuracy(labels, outputs, topk=(1, 5))
            top1 += corr1
            top5 += corr5
            counter += 1
            sub_top1 += corr1
            sub_top5 += corr5
            sub_counter += 1
        print(f"Subset Val:\tTop-1 accuracy: {sub_top1/sub_counter:.5f}\tTop-5 accuracy: {sub_top5/sub_counter:.5f}")
    print(f"Cumulative Val:\nTop-1 accuracy: {top1/counter:.5f}\nTop-5 accuracy: {top5/counter:.5f}")
    return top1/counter, top5/counter

def ppdf(df_F):
    df_F['Label'] = df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
    df_F['Label'] = df_F['Label'].astype(dtype=float)
    df_F['Path'] = "wav/" + df_F['Path']
    return df_F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir", "-d", help="Directory with wav and csv files", default="./Data/")
    args = parser.parse_args()
    DATA_DIR = args.dir
    
    df_meta = pd.read_csv(LOCAL_DATA_DIR + "vox1_meta.csv", sep="\t")
    df_F = pd.read_csv(LOCAL_DATA_DIR + "iden_split.txt", sep=" ", names=["Set", "Path"])
    val_F = pd.read_pickle(LOCAL_DATA_DIR + "val.pkl")
    df_F = ppdf(df_F)
    val_F = ppdf(val_F)

    Datasets = {
        "train": AudioDataset(df_F[df_F['Set'] == 1], DATA_DIR),
        "val": [AudioDataset(val_F[val_F['lengths'] == i], DATA_DIR, is_train=False, batch_size=1) for i in range(300, 1100, 100)],
        "test": AudioDataset(df_F[df_F['Set'] == 3], DATA_DIR, is_train=False, batch_size=1)
    }

    model = VGGM(N_CLASSES)
    loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.SGD(learning_rate=LR, momentum=0.99)
    
    @tf.function
    def train_step(audio, labels):
        with tf.GradientTape() as tape:
            outputs = model(audio, training=True)
            loss = loss_func(labels, outputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, outputs

    best_acc = 75
    best_epoch = 0
    
    for epoch in range(N_EPOCHS):
        model.trainable = True
        running_loss = 0.0
        top1 = 0
        top5 = 0
        loop = tqdm(Datasets['train'])
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        
        for counter, (audio, labels) in enumerate(loop, start=1):
            loss, outputs = train_step(audio, labels)
            running_loss += loss
            corr1, corr5 = accuracy(labels, outputs, topk=(1, 5))
            top1 += corr1
            top5 += corr5
            loop.set_postfix(loss=running_loss.numpy()/(counter), top1_acc=top1/(counter), top5_acc=top5/counter)
        
        model.trainable = False
        acc1, _ = test(model, Datasets['val'])
        if acc1 > best_acc:
            best_acc = acc1
            best_model = model.get_weights()
            best_epoch = epoch
            model.save_weights(os.path.join(MODEL_DIR, f"VGGMVAL_BEST_{best_epoch}_{best_acc:.2f}.h5"))
        
        optimizer.learning_rate.assign(optimizer.learning_rate * (1/1.17))

    print('Finished Training..')
    PATH = os.path.join(MODEL_DIR, "VGGM_F.h5")
    model.save_weights(PATH)
    model.trainable = False
    acc1 = test(model, [Datasets['test']])
