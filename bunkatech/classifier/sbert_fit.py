from sentence_transformers import InputExample, losses, SentenceTransformer
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pickle
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def sentence_pairs_generation(sentences, labels, pairs):
    # initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sentences)):
        currentSentence = sentences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList == label)[0][0]])
        posSentence = sentences[idxB]
        # prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))

        negIdx = np.where(labels != label)[0]
        negSentence = sentences[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))

        # return a 2-tuple of our image pairs and labels
    return pairs


def sbert_fit(
    df,
    col_text="text",
    col_label="president",
    st_model="distiluse-base-multilingual-cased-v1",
    num_training=32,
    num_itr=5,
):

    # Ressources:
    # https://colab.research.google.com/github/MosheWasserb/SetFit/blob/main/SetFit_SST_2.ipynb#scrollTo=aFOzlLAfYOHU
    # https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e

    labels = list(set(df[col_label]))

    # Go further for that line
    df_sample = pd.DataFrame()
    for label in labels:
        dat = df[df[col_label] == label]
        dat = dat.sample(min(num_training, len(dat)))
        df_sample = df_sample.append(dat)

    x_train = df_sample[col_text].values.tolist()
    y_train = df_sample[col_label].values.tolist()

    train_examples = []
    for x in range(num_itr):
        train_examples = sentence_pairs_generation(
            np.array(x_train), np.array(y_train), train_examples
        )

    # S-BERT adaptation
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    model = SentenceTransformer(st_model)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True,
    )

    return model
