import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import get_trec_splits, UNK_TOKEN, PADDING_TOKEN

from tqdm import tqdm



class CNN(nn.Module):
    def __init__(self, embeddings, padding_idx, num_classes):
        super(CNN, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        self.embeddings_size = len(embeddings[0])
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embeddings_size, out_channels=100, kernel_size=k) for k in [3, 4, 5]
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(300, num_classes)
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


def preprocess(sample, text_field, word_to_idx, delim=" "):
    toks = sample[text_field].split(delim)
    sample[text_field] = [word_to_idx.get(tok, word_to_idx[UNK_TOKEN]) for tok in toks]
    return sample


def pad(batch, text_field, label_field, min_length, padding_idx):
    # idx = torch.tensor([sample["idx"] for sample in batch])

    sentences = [sample[text_field] for sample in batch]
    max_len = max(sample.size(0) for sample in sentences)
    if max_len < min_length: # sentence must be at least the length of the longest filter
        max_len = min_length
    sentences = torch.stack([nn.functional.pad(sample, (0, max_len - sample.size(0)), value=padding_idx) for sample in sentences])

    labels = torch.tensor([sample[label_field] for sample in batch])

    return {'sentence': sentences, 'label': labels}


if __name__ == "__main__":

    with open("embeddings.pt", mode="rb") as file:
        embeddings_dict = torch.load(file)
    word_to_idx = embeddings_dict["vocab"]
    embeddings = embeddings_dict["embeddings"]

    train, dev, test = get_trec_splits()

    mapping_fn = lambda sample: preprocess(sample, "text", word_to_idx)
    padding_fn = lambda b: pad(b, "text", "coarse_label",5, word_to_idx[PADDING_TOKEN])

    train = train.map(mapping_fn)
    train.set_format(type="torch")
    train_loader = DataLoader(train, shuffle=True, batch_size=50, collate_fn=padding_fn)

    dev = dev.map(mapping_fn)
    dev.set_format(type="torch")
    dev_loader = DataLoader(dev, batch_size=10, collate_fn=padding_fn)

    test = test.map(mapping_fn)
    test.set_format(type="torch")
    test_loader = DataLoader(test, batch_size=10, collate_fn=padding_fn)

    cnn = CNN(embeddings, padding_idx=word_to_idx[PADDING_TOKEN], num_classes=6)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(cnn.parameters())

    training_loop(train_loader, dev_loader, cnn, loss, optimizer, 3, 1)

    test_acc = evaluate(test_loader, cnn, load_best=True)

    print(f"Test set accuracy: {test_acc}")