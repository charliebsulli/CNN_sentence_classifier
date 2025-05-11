from datasets import load_dataset
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm

UNK_TOKEN = "<unk>"
PADDING_TOKEN = "<pad>"

class Convolution(nn.Module):
    def __init__(self, window_size, num_filters, embeddings_size):
        super(Convolution, self).__init__()
        self.window_size = window_size
        self.embeddings_size = embeddings_size
        self.flatten = nn.Flatten()
        self.filter = nn.Linear(self.window_size * self.embeddings_size, num_filters)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        # concatenate vectors in input_seq
        sentences = self.flatten(input_seq)
        window = self.window_size * self.embeddings_size

        # create tensor of segments of the sentence
        slices = []
        for i in np.arange(0, len(sentences[0]) + self.embeddings_size - window, self.embeddings_size):
            next_slice = sentences[:, i:i+window]
            slices.append(next_slice)
        slices = torch.stack(slices, dim=1).float()

        feature_map = torch.transpose(self.relu(self.filter(slices)), 1, 2)

        return feature_map # a stack of (num_filters) feature maps for each batch


class CNN(nn.Module):
    def __init__(self, embeddings, padding_idx):
        super(CNN, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        self.embeddings_size = len(embeddings[0])
        self.conv3 = Convolution(3, 100, self.embeddings_size) # TODO try nn.Conv1d
        self.conv4 = Convolution(4, 100, self.embeddings_size)
        self.conv5 = Convolution(5, 100, self.embeddings_size)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embeddings_size, out_channels=100, kernel_size=k) for k in [3, 4, 5]
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(300, 2)
        # self.softmax = nn.Softmax()

    def forward(self, input_seq):
        # embed input sequence
        embedded = self.embeddings(input_seq)
        embedded = torch.transpose(embedded, dim0=1, dim1=2) # in-channels should be dimension after batch

        # convolution layer
        feature_map = [self.relu(conv(embedded)) for conv in self.convs]

        # max pooling
        # shape depends on sequence length + filter size, so we call it as needed
        pooled_features = [torch.squeeze(torch.nn.functional.max_pool1d(feats, feats.shape[2]), dim=2) for feats in feature_map]


        # compute + max-pool feature maps
        # three_feats = torch.max(self.conv3(embedded), -1).values
        # four_feats = torch.max(self.conv4(embedded), -1).values
        # five_feats = torch.max(self.conv5(embedded), -1).values

        # concatenate features
        features = torch.cat(pooled_features, 1)

        # softmax layer
        logits = self.linear(self.dropout(features))

        return logits


def training_loop(train_dataloader, dev_dataloader, model, loss_fn, optimizer, lambd, epochs):
    model.train()
    best_dev_acc = 0
    for e in range(epochs):
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            pred = model(batch['sentence'])
            loss = loss_fn(pred, batch['label']) # dim of input must be 1 greater than dim of target

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # regularization
            with torch.no_grad():
                new_weights = []
                for w in model.linear.weight.data:
                    norm = torch.linalg.norm(w)
                    if norm > lambd/2:
                        # rescale w by (lambda/2) / norm
                        new_w = ((lambd/2) / norm) * w
                        new_weights.append(new_w)
                    else:
                        new_weights.append(w)
                ws = torch.stack(new_weights)
                model.linear.weight.data = ws


        print(f"Epoch: {e}")
        dev_acc = evaluate(dev_dataloader, model)
        print(f"Dev set accuracy: {dev_acc}")
        if dev_acc > best_dev_acc:
            print("Saving new best model...")
            torch.save(model.state_dict(), "best_model.pt")
            best_dev_acc = dev_acc


def evaluate(dataloader, model, load_best = False):
    if load_best:
        print("Loading best model...")
        model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logits = model(batch['sentence'])
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            for gold_label, pred_label in zip(batch['label'], pred):
                if gold_label == pred_label:
                    correct += 1
                total += 1

    return correct / total


def preprocess(sample, word_to_idx):
    toks = sample["sentence"].split(" ")
    sample["sentence"] = [word_to_idx.get(tok, word_to_idx[UNK_TOKEN]) for tok in toks]
    return sample


def pad(batch, min_length, padding_idx):
    idx = torch.tensor([sample["idx"] for sample in batch])

    sentences = [sample["sentence"] for sample in batch]
    max_len = max(sample.size(0) for sample in sentences)
    if max_len < min_length: # sentence must be at least the length of the longest filter
        max_len = min_length
    sentences = torch.stack([nn.functional.pad(sample, (0, max_len - sample.size(0)), value=padding_idx) for sample in sentences])

    labels = torch.tensor([sample["label"] for sample in batch])

    return {'idx': idx, 'sentence': sentences, 'label': labels}


if __name__ == "__main__":

    with open("embeddings.pt", mode="rb") as file:
        embeddings_dict = torch.load(file)
    word_to_idx = embeddings_dict["vocab"]
    embeddings = embeddings_dict["embeddings"]

    padding_fn = lambda b: pad(b, 5, word_to_idx[PADDING_TOKEN])
    # load SST-2 dataset, mapping tokens in sentences to indices
    train = load_dataset("stanfordnlp/sst2", split="train").map(lambda sample: preprocess(sample, word_to_idx))
    train.set_format(type="torch")
    train_loader = DataLoader(train, shuffle=True, batch_size=50, collate_fn=padding_fn)

    dev = load_dataset("stanfordnlp/sst2", split="validation").map(lambda sample: preprocess(sample, word_to_idx))
    dev.set_format(type="torch")
    dev_loader = DataLoader(dev, batch_size=10, collate_fn=padding_fn)

    # test = load_dataset("stanfordnlp/sst2", split="test").map(lambda sample: preprocess(sample, word_to_idx))
    # test.set_format(type="torch")
    # test_loader = DataLoader(test, batch_size=10, collate_fn=padding_fn)

    cnn = CNN(embeddings, padding_idx=word_to_idx[PADDING_TOKEN])

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(cnn.parameters()) # TODO hyperparams

    training_loop(train_loader, dev_loader, cnn, loss, optimizer, 3, 25)

    dev_acc = evaluate(dev_loader, cnn, load_best=True)

    print(f"Dev set accuracy: {dev_acc}")