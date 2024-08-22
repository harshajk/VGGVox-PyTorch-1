#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tqdm.auto import tqdm
from vggm import VGGM
import argparse
from train import AudioDataset, accuracy, ppdf, LOCAL_DATA_DIR, MODEL_DIR

def test(model, dataset):
    corr1 = 0
    corr5 = 0
    counter = 0
    top1 = 0
    top5 = 0
    for audio, labels in dataset:
        outputs = model(audio, training=False)
        corr1, corr5 = accuracy(labels, outputs)
        top1 += corr1
        top5 += corr5
        counter += 1
    print(f"\nTop-1 accuracy: {top1/counter:.5f}\nTop-5 accuracy: {top5/counter:.5f}")
    return top1/counter, top5/counter

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir", "-d", help="Directory with wav and csv files", default="./Data/")
    args = parser.parse_args()
    DATA_DIR = args.dir 
    df_meta = pd.read_csv(LOCAL_DATA_DIR+"vox1_meta.csv", sep="\t")
    df_F = pd.read_csv(LOCAL_DATA_DIR+"iden_split.txt", sep=" ", names=["Set","Path"])
    df_F = ppdf(df_F)
    
    Datasets = {
        "val": AudioDataset(df_F[df_F['Set']==2], DATA_DIR, is_train=False, batch_size=1),
        "test": AudioDataset(df_F[df_F['Set']==3], DATA_DIR, is_train=False, batch_size=1)
    }
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU available")
    else:
        print("Using CPU")
    
    model = VGGM(1251)
    model.build((None, 48320, 1))  # Adjust the input shape if necessary
    model.load_weights(MODEL_DIR+"VGGM300_BEST_140_81.99.h5")
    
    print("\nVal Score:\n")
    acc1, _ = test(model, Datasets['val'])
        
    print("\nTest Score:\n")
    acc1 = test(model, Datasets['test'])
