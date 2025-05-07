import os.path
import pickle

import gensim.downloader
import numpy as np
import torch
from datasets import load_dataset

# given the SST-2 dataset, save word2vec vectors for that dataset to a file
def save_embeddings(dataset, model, idx_dict_path, embeddings_path): # maybe pass path to idx dict, embeddings as well
    # get vocab of the dataset
    vocab = set()
    for sample in dataset:
        vocab.update(sample["sentence"].split(" "))

    word_to_idx = {}
    embeddings = []
    for token in vocab:
        word_to_idx[token] = len(embeddings)
        try:
            embeddings.append(model[token])
        except KeyError:
            # word is OOV embeddings
            embeddings.append(np.zeros(300))

    with open(idx_dict_path, mode="wb") as file:
        pickle.dump(word_to_idx, file)

    embedding_tensor = torch.tensor(embeddings)
    torch.save(embedding_tensor, embeddings_path)

    return vocab


def look_up_embedding(token, idx_dict_path, embeddings_path):
    with open(idx_dict_path, mode="rb") as file:
        word_to_idx = pickle.load(file)

    embeddings = torch.load(embeddings_path)

    return embeddings[word_to_idx[token]]


if __name__ == "__main__":
    train = load_dataset("stanfordnlp/sst2", split="train")
    word2vec = gensim.downloader.load("word2vec-google-news-300")
    idx_path = "word_to_idx.pkl"
    embeddings_path = "embeddings.pt"
    save_embeddings(train, word2vec, idx_path, embeddings_path)


