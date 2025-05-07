import pickle

from datasets import load_dataset
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm


class Convolution(nn.Module):
    def __init__(self, window_size, num_filters, embeddings_size):
        super(Convolution, self).__init__()
        self.window_size = window_size
        self.embeddings_size = embeddings_size
        self.flatten = nn.Flatten(0, -1) # TODO change once batch sizes implemented
        self.filter = nn.Linear(self.window_size * self.embeddings_size, num_filters)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        # concatenate vectors in input_seq
        sentence = self.flatten(input_seq)
        window = self.window_size * self.embeddings_size

        # equivalent to adding a word to each end of the sentence until long enough for
        # an application of the filter
        # TODO may need to make this more precise
        while len(sentence) < window:
            sentence = nn.functional.pad(sentence, (self.embeddings_size, self.embeddings_size))

        # create matrix of segments of the sentence
        slices = []
        # steps one word at a time over the rest of the sentence
        for i in np.arange(0, len(sentence) + self.embeddings_size - window, self.embeddings_size):
            next_slice = sentence[i:i+window]
            slices.append(next_slice)
        slices = torch.stack(slices).float()
        feature_map = torch.transpose(self.relu(self.filter(slices)), 0, 1)

        return feature_map # a stack of (num_filters) feature maps


class CNN(nn.Module):
    def __init__(self, embeddings):
        super(CNN, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.conv3 = Convolution(3, 100, 300)
        self.conv4 = Convolution(4, 100, 300)
        self.conv5 = Convolution(5, 100, 300)
        self.softmax_layer = nn.Sequential(
            nn.Linear(300, 2),
            nn.Softmax()
        )

    def forward(self, input_seq):
        # embed input sequence
        embedded = self.embeddings(input_seq)

        # compute + max-pool feature maps
        three_feats = torch.max(self.conv3(embedded), 1).values
        four_feats = torch.max(self.conv4(embedded), 1).values
        five_feats = torch.max(self.conv5(embedded), 1).values

        # concatenate features
        features = torch.cat((three_feats, four_feats, five_feats), 0)

        # softmax layer
        probs = self.softmax_layer(features)

        return probs


def train(dataset, model, loss_fn, optimizer, word_to_idx):
    model.train()
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)): # is there some kind of batched way to perform this lookup?
        # process the sample
        X, y = process_sample(sample)

        pred = model(X) # pass through model
        loss = loss_fn(pred, torch.tensor(y))

        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(dataset, model, word_to_idx):
    model.eval()
    correct = 0
    total = len(dataset)
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            X, y = process_sample(sample)
            pred = np.argmax(model(X))
            if pred == y:
                correct += 1

    return correct / total


def process_sample(sample):
    # translates the sentence from the SST2 dataset into indices, saves class label
    tokens = sample["sentence"].split(" ")
    X = []
    for token in tokens:
        if token in word_to_idx:
            X.append(word_to_idx[token])
        else:
            X.append(0) # just a hack to test training loop, fix how I am loading the data
    y = sample['label']

    return torch.tensor(X), y


if __name__ == "__main__":
    # load SST-2 dataset
    train_set = load_dataset("stanfordnlp/sst2", split="train")
    dev = load_dataset("stanfordnlp/sst2", split="validation")
    test = load_dataset("stanfordnlp/sst2", split="test")


    # get saved embeddings
    embeddings = torch.load("embeddings.pt")

    # get word to index dict
    with open("word_to_idx.pkl", mode="rb") as file:
        word_to_idx = pickle.load(file)

    # train_dataloader = DataLoader(train_set, batch_size=1) # later add capability to model for bigger batches

    cnn = CNN(embeddings)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(cnn.parameters()) # TODO hyperparams

    train(train_set, cnn, loss, optimizer, word_to_idx)

    dev_acc = evaluate(dev, cnn, word_to_idx)

    print(f"Dev set accuracy: {dev_acc}")