import os
import sys
import math
import torch
import torch.utils.data
from torch import nn
from unidecode import unidecode
import tqdm
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleanMethod import giveData

def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


_ = torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### DATASETS:
###  -> data_Twitter
###  -> data_IMDB
###  -> data_RottenTomatoes
CHOOSE_DATASET = "data_Twitter"
BATCH_SIZE = 128
NUM_LAYER = 3
NUM_QUBITS = 6
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

test_size = 0.1
print(f"Dividing data into Train/Test - test size ratio = {test_size}")
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    range(len(target_sentiment)), 
    test_size=test_size, 
    shuffle=True, 
    stratify=target_sentiment,
    random_state=42
)

batch_size = 64
train_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_sentences[train_idx], target_sentiment[train_idx]), batch_size, drop_last=True)
test_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_sentences[test_idx], target_sentiment[test_idx]), batch_size, drop_last=True)

print("Split data successful")
print(f"Train size: {len(train_dataset)} batches")
print(f"Test size: {len(test_dataset)} batches\n")

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x



n_qubits = NUM_QUBITS
num_layers = NUM_QUBITS

print("layers =",num_layers)
print("num_qubits =",n_qubits)

vocab_size = dic_length
embed_size = 2**n_qubits
hidden_size = 2**n_qubits
output_size = num_sentiment
dropout = 0
learning_rate = LEARNING_RATE


model = SentimentLSTM(vocab_size, embed_size, hidden_size, output_size, num_layers, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

time_start_training = time.time()

def train_model(model, train_dataset, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        
        time_start_epoch = time.time()
        
        for sentences, labels in tqdm.tqdm(train_dataset):
            sentences, labels = sentences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy * 100:.2f}%')
        # Evaluating the model
        test_accuracy = evaluate_model(model, test_dataset)
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

def evaluate_model(model, test_dataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentences, labels in tqdm.tqdm(test_dataset):
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = model(sentences)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

train_model(model, train_dataset, optimizer, criterion, num_epochs=15)
