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


def training_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    print(next(iter(dataloader)))
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        pred = model(batch['sentence'])
        loss = loss_fn(pred, batch['label'][0]) # dim of input must be 1 greater than dim of target

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(dataloader, model):
    model.eval()
    correct = 0
    total = len(dataloader) # keep batch size to 1
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            pred = np.argmax(model(sample['sentence']))
            if pred == sample['label'][0]:
                correct += 1

    return correct / total


# this means my approach to unknown tokens is to map them all to the same token
# in order to do this, need to initialize a random unk embedding
# when I create the embeddings
def preprocess(sample, word_to_idx):
    toks = sample["sentence"].split(" ")
    sample["sentence"] = [word_to_idx.get(tok, word_to_idx["the"]) for tok in toks] # TODO update unk token
    return sample


if __name__ == "__main__":

    # get saved word to index mapping
    # TODO if it does not exist, create it
    with open("word_to_idx.pkl", mode="rb") as file:
        word_to_idx = pickle.load(file)

    # load SST-2 dataset, mapping sentences to indices TODO can parallelize mapping
    train = load_dataset("stanfordnlp/sst2", split="train").map(lambda sample: preprocess(sample, word_to_idx))
    train.set_format(type="torch")
    train_loader = DataLoader(train, batch_size=1) # TODO update batch size

    dev = load_dataset("stanfordnlp/sst2", split="validation").map(lambda sample: preprocess(sample, word_to_idx))
    dev.set_format(type="torch")
    dev_loader = DataLoader(dev, batch_size=1)
    # test = load_dataset("stanfordnlp/sst2", split="test")

    # get saved embeddings
    embeddings = torch.load("embeddings.pt")

    cnn = CNN(embeddings)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(cnn.parameters()) # TODO hyperparams

    training_loop(train_loader, cnn, loss, optimizer)

    dev_acc = evaluate(dev_loader, cnn)

    print(f"Dev set accuracy: {dev_acc}")