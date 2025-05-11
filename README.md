# CNN for sentence classification
Re-implementation of CNN-non-static from [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).
I wrote this to practice using PyTorch. It is written to use the [TREC](https://huggingface.co/datasets/CogComp/trec) dataset,
but requires little modification to use with the other datasets from the paper.

## Setup
1. Clone repository
2. Install necessary python packages (torch, datasets, gensim)
3. Run utils.py to save embeddings and create a mapping from tokens to indices
4. Run cnn.py to train and evaluate the model

Accuracy on [TREC](https://huggingface.co/datasets/CogComp/trec): 90.20

## Citations

    @inproceedings{kim_convolutional_2014,
        address = {Doha, Qatar},
        title = {Convolutional {Neural} {Networks} for {Sentence} {Classification}},
        url = {https://aclanthology.org/D14-1181/},
        doi = {10.3115/v1/D14-1181},
        urldate = {2025-05-06},
        booktitle = {Proceedings of the 2014 {Conference} on {Empirical} {Methods} in {Natural} {Language} {Processing} ({EMNLP})},
        publisher = {Association for Computational Linguistics},
        author = {Kim, Yoon},
        editor = {Moschitti, Alessandro and Pang, Bo and Daelemans, Walter},
        month = oct,
        year = {2014},
        keywords = {read},
        pages = {1746--1751},
    }

    @inproceedings{li-roth-2002-learning,
        title = "Learning Question Classifiers",
        author = "Li, Xin  and
          Roth, Dan",
        booktitle = "{COLING} 2002: The 19th International Conference on Computational Linguistics",
        year = "2002",
        url = "https://www.aclweb.org/anthology/C02-1150",
    }

    @inproceedings{hovy-etal-2001-toward,
        title = "Toward Semantics-Based Answer Pinpointing",
        author = "Hovy, Eduard  and
          Gerber, Laurie  and
          Hermjakob, Ulf  and
          Lin, Chin-Yew  and
          Ravichandran, Deepak",
        booktitle = "Proceedings of the First International Conference on Human Language Technology Research",
        year = "2001",
        url = "https://www.aclweb.org/anthology/H01-1069",
    }
