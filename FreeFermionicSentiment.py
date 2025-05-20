import qiskit

import os
import sys
import shutil
import random
from string import ascii_letters
import numpy

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from unidecode import unidecode
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import FFQuantumDevice

from cleanMethod import giveData

def read_lines_into_list(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    return [int(line) for line in lines[1:]]

_ = torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### DATASETS:
###  -> data_Twitter
###  -> data_IMDB
###  -> data_RottenTomatoes
CHOOSE_DATASET = "data_Twitter"
BATCH_SIZE = 128
NUM_LAYER = 1
NUM_QUBITS = 4
EPOCHS = 10
LEARNING_RATE = 0.005

data_dir = f"./data/{CHOOSE_DATASET}/"
dataset_name = data_dir.split("/")[-2]

sent2label = {
    file_name.split(".")[0]: torch.tensor(i, dtype=torch.long)
    for i, file_name in enumerate(os.listdir(data_dir))
}

print("\nSentiment classes:")
for sent in sent2label:
    print(sent)
print()

num_sentiment = len(sent2label)

remove_non_english_words = True

print(f"Building dictionary -> remove non english words: {remove_non_english_words}")
all_categories, category_lines, dictionary = giveData(data_dir,remove_non_english_words)
dic_length = len(dictionary)
print(f"Dictionary build of size {dic_length}\n")

def word2index(word):
    if word in dictionary:
        return dictionary[word]
    else: return 0

def sentence2tensor(sentence, num_words):
    tensor = torch.zeros(num_words, dtype=torch.long)
    split_sent = sentence.split()[:num_words]
    for i, word in enumerate(split_sent):
        tensor[i] = word2index(word)
    return tensor

print("Parsing data")
tensor_sentences = []
target_sentiment = []

if dataset_name == "data_IMDB":
    pad_size = 100
elif dataset_name == "data_RottenTomatoes":
    pad_size = 40
elif dataset_name == "data_Twitter":
    pad_size = 33

for file in os.listdir(data_dir):
    with open(os.path.join(data_dir, file)) as f:
        lang = file.split(".")[0]
        names = [unidecode(line.rstrip()) for line in f]
        for name in names:
            try:
                tensor_sentences.append(sentence2tensor(name, pad_size))
                target_sentiment.append(sent2label[lang])
            except KeyError:
                pass
print(f"Data imported successful, and padded to length = {pad_size}\n")

tensor_sentences = torch.stack(tensor_sentences)
target_sentiment = torch.stack(target_sentiment)

test_size=0.1
print(f"Dividing data into Train/Test - test size ratio = {test_size}")
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    range(len(target_sentiment)), 
    test_size=test_size, 
    shuffle=True, 
    stratify=target_sentiment
)

batch_size = BATCH_SIZE
train_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_sentences[train_idx], target_sentiment[train_idx]), batch_size, drop_last=True)
test_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_sentences[test_idx], target_sentiment[test_idx]), batch_size, drop_last=True)

print("Split data successful")
print(f"Train size: {len(train_dataset)} batches")
print(f"Test size: {len(test_dataset)} batches\n")

num_layer = NUM_LAYER
num_qubits = NUM_QUBITS
num_angles = num_layer * (2 * num_qubits - 1)
number_words = dic_length
num_class = num_sentiment
learning_rate = LEARNING_RATE

    
class RNN(torch.nn.Module):
    def __init__(self, input_size, num_angles):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.L = torch.nn.Linear(self.input_size, num_angles)

    def forward(self, input, circuit:FFQuantumDevice):
        input = torch.nn.functional.one_hot(input, self.input_size).to(torch.float)
        #print(input.shape)
        angles = self.L(input)

        for _ in range(num_layer):
            ang = 0
            circuit.rxx_layer(angles[:,ang:ang+num_qubits-1])
            ang += num_qubits - 1
            circuit.rz_layer(angles[:,ang:ang+num_qubits])
            ang += num_qubits

        return circuit

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Lclass = torch.nn.Linear(num_qubits, num_class)


    def forward(self, circuit:FFQuantumDevice):
        meas = circuit.z_exp_all()
        meas_shape = self.Lclass(meas)
        res = nn.Softmax(dim=-1)
        soft = res(meas_shape)
        return soft

model = RNN(number_words, num_angles)
classifier = Classifier()
optim = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def evaluate(model, classifier, inputs):
    hidden = FFQuantumDevice(num_qubits, batch_size, device=device)
    for i in range(inputs.shape[1]):
        input = inputs[:,i]
        hidden = model.forward(input, hidden)
    
    output = classifier.forward(hidden)
    return output

def calculate_accuracy(model, classifier, dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence_tensor, true_output in tqdm.tqdm(dataset):
            output = evaluate(model, classifier, sentence_tensor)
            _, predicted = torch.max(output, 1)
            total += true_output.size(0)
            correct += (predicted == true_output).sum().item()
    return correct / total


import time 
import math
def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

start = time.time()
for epoch in range(EPOCHS):
    correct = 0
    total = 0
    for i, (sentence_tensor, true_output) in enumerate(tqdm.tqdm(train_dataset)):
        optim.zero_grad()
        output = evaluate(model, classifier, sentence_tensor)
        loss = criterion(output, true_output)
        loss.backward()
        optim.step()

        _, predicted = torch.max(output, 1)

        total += true_output.size(0)
        correct += (predicted == true_output).sum().item()

    # Calculate accuracy on test dataset
    train_accuracy = correct / total
    print(f'Epoch: {epoch} Train Accuracy: {train_accuracy * 100:.2f}% - time start: {timeSince(start)}')

    # Calculate accuracy on test dataset
    test_accuracy = calculate_accuracy(model, classifier, test_dataset)
    print(f'Epoch: {epoch} Test Accuracy: {test_accuracy * 100:.2f}% - time start: {timeSince(start)}')
