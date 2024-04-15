from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchaudio
from torchaudio import transforms
import os
import random
import torch.nn.functional as F
from torch.nn import init
from torch import nn
import shutil

from model import *


model = AudioClassifier()
model.load_state_dict(torch.load("./model.pt"))
model.eval()


def prepare_file(audio_file):
    sig, sr = torchaudio.load(audio_file)
    spec = transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)

    return spec.unsqueeze(0)


def get_prediction(audio_file):
    inputs = prepare_file(audio_file)
    output = model.forward(inputs)
    normalized_probs = F.softmax(output, dim=1)
    max_prob, class_index = torch.max(normalized_probs, dim=1)
    confidence = max_prob.item()
    predicted_class = class_index.item()
    return confidence, predicted_class


def is_angry(audio_file):
    _, angry = get_prediction(audio_file)
    return angry == 0


angry_folder_path = "./data/angry"
angry_files = [os.path.join(angry_folder_path, file) for file in os.listdir(
    angry_folder_path) if file.lower().endswith(".wav")]

not_angry_folder_path = "./data/not-angry"
not_angry_files = [os.path.join(not_angry_folder_path, file) for file in os.listdir(
    not_angry_folder_path) if file.lower().endswith(".wav")]


if os.path.exists('./data/eval'):
    shutil.rmtree('./data/eval')

os.makedirs('./data/eval/false-positives')
os.makedirs('./data/eval/false-negatives')


print('finding false-negatives...')
for angry in angry_files:
    if not is_angry(angry):
        print(f"{angry} {get_prediction(angry)}")
        shutil.copy(angry, './data/eval/false-negatives')


print('finding false-positives...')
for angry in not_angry_files:
    if is_angry(angry):
        print(f"{angry} {get_prediction(angry)}")
        print(angry)
        shutil.copy(angry, './data/eval/false-positives')


print("angry")
for i in range(10):
    file = random.choice(angry_files)
    print(f"{file} {get_prediction(file)}")

print("\n\n\nnot angry")
for i in range(10):
    file = random.choice(not_angry_files)
    print(f"{file} {get_prediction(file)}")
