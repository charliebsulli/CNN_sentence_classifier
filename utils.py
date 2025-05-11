import gensim.downloader
import torch
from datasets import load_dataset

from cnn import UNK_TOKEN, PADDING_TOKEN


def get_vocab(dataset):
    vocab = set()
    for sample in dataset:
        vocab.update(sample["sentence"].split(" "))
    vocab_dict = {tok: i+2 for i, tok in enumerate(vocab)}
    vocab_dict[PADDING_TOKEN] = 0
    vocab_dict[UNK_TOKEN] = 1
    return vocab_dict


def get_embeddings(vocab, model, embeddings_size):
    embeddings = torch.zeros((len(vocab), embeddings_size))
    for tok, idx in vocab.items():
        try:
            embeddings[idx] = torch.tensor(model[tok])
        except KeyError: # OOV embedding
            pass
    return embeddings


if __name__ == "__main__":
    train = load_dataset("stanfordnlp/sst2", split="train")
    word2vec = gensim.downloader.load("word2vec-google-news-300")
    vocab = get_vocab(train)
    embeddings = get_embeddings(vocab, word2vec, 300)
    embeddings_dict = {"vocab": vocab, "embeddings": embeddings}
    torch.save(embeddings_dict, "embeddings.pt")

