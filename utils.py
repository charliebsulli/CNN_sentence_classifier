import gensim.downloader
import torch
from datasets import load_dataset

from cnn import UNK_TOKEN, PADDING_TOKEN


def get_vocab(dataset, text_field, delim=" "):
    vocab = set()
    for sample in dataset:
        vocab.update(sample[text_field].split(delim))
    vocab_dict = {tok: i+2 for i, tok in enumerate(vocab)}
    vocab_dict[PADDING_TOKEN] = 0
    vocab_dict[UNK_TOKEN] = 1
    return vocab_dict


def get_embeddings(vocab, model, embeddings_size):
    embeddings = torch.rand((len(vocab), embeddings_size))
    for tok, idx in vocab.items():
        try:
            embeddings[idx] = torch.tensor(model[tok])
        except KeyError: # OOV embedding
            pass
    embeddings[vocab[PADDING_TOKEN]] = torch.zeros(embeddings_size)
    return embeddings


if __name__ == "__main__":
    train = load_dataset("CogComp/trec", split="train")
    ds = train.train_test_split(test_size=0.1, shuffle=True, seed=2)
    train = ds["train"]
    dev = ds["test"]
    word2vec = gensim.downloader.load("word2vec-google-news-300")
    vocab = get_vocab(train, "text")
    embeddings = get_embeddings(vocab, word2vec, 300)
    embeddings_dict = {"vocab": vocab, "embeddings": embeddings}
    torch.save(embeddings_dict, "embeddings.pt")

