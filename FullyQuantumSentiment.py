
import os

import torch
import torch.utils.data
from torch import nn
import torchquantum as tq
from unidecode import unidecode
import tqdm

from cleanMethod import giveData

_ = torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### DATASETS:
###  -> data_Twitter
###  -> data_IMDB
###  -> data_RottenTomatoes
CHOOSE_DATASET = "data_Twitter"
BATCH_SIZE = 128
NUM_LAYER = 2
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
    #print(split_sent)
    for i, word in enumerate(split_sent):
        #print(word)
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

batch_size = 256
train_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_sentences[train_idx], target_sentiment[train_idx]), batch_size, drop_last=True)
test_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_sentences[test_idx], target_sentiment[test_idx]), batch_size, drop_last=True)

print("Split data successful")
print(f"Train size: {len(train_dataset)} batches")
print(f"Test size: {len(test_dataset)} batches\n")



num_layer = NUM_LAYER
num_qubits = NUM_QUBITS
print("Layer =",num_layer)
print("Qubits =",num_qubits)
num_angles = num_layer * (4 * num_qubits - 1)
number_words = dic_length
num_class = num_sentiment
learning_rate = LEARNING_RATE
EPOCHS = EPOCHS
if num_qubits > 7:
    EPOCHS = 5

WRITE_BOOL = True

if dataset_name == "data_IMDB":
    filename = "../Results/QuantumSentiment/QuantumIMDB.csv"
elif dataset_name == "data_RottenTomatoes":
    filename = "../Results/QuantumSentiment/QuantumRottenTomatoes.csv"
elif dataset_name == "data_Twitter":
    filename = "../Results/QuantumSentiment/QuantumTwitter.csv"

filename = "../Results/QuantumSentiment/FQincreaseQubitsTwitter.csv"

def add_matchgate(qdev:tq.QuantumDevice, angles):
    ang = 0
    for i in range(num_qubits-1):
        qdev.rxx(params=angles[:, ang], wires=[i, i+1])
        ang += 1
        qdev.u3(params=angles[:, ang:ang+3], wires=i)
        ang += 3
    qdev.u3(params=angles[:, ang:ang+3], wires=num_qubits-1)

class RNN(torch.nn.Module):
    def __init__(self, input_size, num_angles):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.L = torch.nn.Linear(self.input_size, num_angles)

    def forward(self, input, qdev:tq.QuantumDevice):
        input = torch.nn.functional.one_hot(input, self.input_size).to(torch.float)
        angles = self.L(input)

        for lay in range(num_layer):
            add_matchgate(qdev,angles[:,lay * (4 * num_qubits - 1) : (lay+1) * (4 * num_qubits - 1)])
        return qdev

class Classifier(torch.nn.Module):
    def __init__(self,num_qubits):
        super().__init__()
        self.num_qubits = num_qubits
        self.Lclass = torch.nn.Linear(num_qubits, num_class)

    def forward(self, qdev:tq.QuantumDevice):
        class_list = []
        for i in range(self.num_qubits):
            meas = "I"*i + "Z" + "I"*(self.num_qubits - i - 1)
            class_list.append(tq.measurement.expval_joint_analytical(qdev, meas))
        res = nn.Softmax(dim=-1)
        stack = torch.stack(class_list, dim=-1)
        soft = res(stack)
        classLayer = self.Lclass(soft)
        return classLayer

model = RNN(number_words, num_angles)
classifier = Classifier(num_qubits)
optim = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def evaluate(model, classifier, inputs):
    hidden = tq.QuantumDevice(num_qubits, bsz=batch_size)
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
    start_epoch = time.time()
    for i, (sentence_tensor, true_output) in enumerate(tqdm.tqdm(train_dataset)):
        optim.zero_grad()
        output = evaluate(model, classifier, sentence_tensor)
        loss = criterion(output, true_output)
        loss.backward()
        optim.step()

        _, predicted = torch.max(output, 1)
        total += true_output.size(0)
        correct += (predicted == true_output).sum().item()

    train_accuracy = correct / total
    print(f'Epoch: {epoch} Train Accuracy: {train_accuracy * 100:.2f}% - time start: {timeSince(start)}')

    test_accuracy = calculate_accuracy(model, classifier, test_dataset)
    print(f'Epoch: {epoch} Test Accuracy: {test_accuracy * 100:.2f}% - time start: {timeSince(start)}')