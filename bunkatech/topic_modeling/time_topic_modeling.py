import pandas as pd
import warnings
from sklearn.cluster import KMeans
from extract_terms import extract_terms_df
from indexer import indexer
from get_embeddings import get_embeddings
from make_bubble import make_bubble
from specificity import specificity
from sklearn.cluster import OPTICS
from sklearn.datasets import fetch_20newsgroups
from random import sample
import numpy as np
import logging
import umap
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s : %(message)s", level=logging.INFO
)


def cosine_distance_exponential_time_decay(bert_vectors, timestamps, temperature=1):
    """Compute embeddings with temporal adjustment"""

    # compute distances between embeddings
    cosine_distance_matrix = pairwise_distances(bert_vectors, metric="cosine")
    # We use l2 since the implementation for pairwise_distances() is much faster than our custom (x-y)²

    # compute distances between timestamps
    timematrix = pairwise_distances(np.array(timestamps).reshape(-1, 1), metric="l2")
    timematrix_renorm = timematrix / np.max(timematrix)
    exp_timematrix = np.exp(-timematrix_renorm / temperature)

    # Join the two embeddings
    final_emb = cosine_distance_matrix * exp_timematrix

    return final_emb


warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


def reduction(distance_matrix, n_components_reduc=5):
    logging.info("UMAP Reduction...")
    X_embedded_fit = umap.UMAP(
        n_components=n_components_reduc,
        metric="precomputed",
    ).fit_transform(distance_matrix)
    return X_embedded_fit


def topic_modeling(
    docs: list,
    timestamps: list,
    topic_number=20,
    temperature=10,
    n_components_reduc=5,
    top_terms=20,
    ngrams=(2, 2),
    ents=False,
    sample_terms=2000,
    language="en",
    embedding_path="embeddings/nyt-russia-headlines-post2014_sentence-bert.npy",
):

    # Embeddings & UMAP

    # Load embeddings if they have already been computed
    if embedding_path is not None:
        print("Load Embedding...")
        embeddings = np.load(embedding_path)
    else:
        print("Create Embeddings...")
        model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        print("Territory embedding..")
        embeddings = model.encode(docs, show_progress_bar=True)
        np.save(
            "embeddings/nyt-russia-headlines-post2014_sentence-bert-sampled.npy",
            embeddings,
        )

    print("Create Time Decay...")
    # Add the time decay method
    final_emb = cosine_distance_exponential_time_decay(
        bert_vectors=embeddings, timestamps=timestamps, temperature=temperature
    )

    print("Umap reduction...")
    embeddings_reduced = reduction(final_emb, n_components_reduc=n_components_reduc)

    # Create dataframe
    df = pd.DataFrame(docs, columns=["text_var"])

    # clusters
    print("Create Clusters...")
    df["cluster"] = (
        KMeans(n_clusters=topic_number).fit(embeddings_reduced).labels_.astype(str)
    )

    topic_size = df.groupby("cluster").count().reset_index()
    topic_size.columns = ["cluster", "topic_size"]

    print("Extract Terms...")
    # Extract Terms
    terms = extract_terms_df(
        df,
        text_var="text_var",
        limit=2000,
        sample_size=sample_terms,
        ngs=True,  # ngrams
        ents=ents,  # entities
        ncs=False,  # nouns
        drop_emoji=True,
        remove_punctuation=False,
        ngrams=ngrams,
        include_pos=["NOUN", "PROPN", "ADJ"],
        include_types=["PERSON", "ORG"],
        language=language,
    )

    # Index with the list of original words
    df_terms = terms.copy()
    df_terms["text"] = df_terms["text"].apply(lambda x: x.split(" | "))
    df_terms = df_terms.explode("text").reset_index(drop=True)
    list_terms = df_terms["text"].tolist()

    df_indexed = indexer(df["text_var"].tolist(), list_terms, db_path=".")
    # get the Main form and the lemma
    df_indexed_full = pd.merge(df_indexed, df_terms, left_on="words", right_on="text")
    df_indexed_full = df_indexed_full[["docs", "lemma", "main form", "text"]].copy()

    data = pd.merge(df_indexed_full, df, left_on="docs", right_on="text_var")
    data = data.drop("docs", axis=1)

    _, _, edge = specificity(data, X="cluster", Y="main form", Z=None, top_n=top_terms)

    topics = (
        edge.groupby("cluster")["main form"]
        .apply(lambda x: " | ".join(x))
        .reset_index()
    )

    final = pd.merge(topics, topic_size, on="cluster")
    final = final.sort_values("topic_size", ascending=False)
    final = final.reset_index(drop=True)

    return final


if __name__ == "__main__":

    data = pd.read_csv("data/nyt-russia-headlines-post2014-sampled.csv")

    docs = data["lead"].to_list()
    timestamps = data["timestamp"].to_list()

    temperature = 10

    topics = topic_modeling(
        docs,
        timestamps,
        topic_number=20,
        n_components_reduc=5,
        temperature=temperature,
        top_terms=50,
        ngrams=(2, 2),
        ents=False,
        sample_terms=2000,
        language="en",
        embedding_path="embeddings/nyt-russia-headlines-post2014_sentence-bert-sampled.npy",
    )

    topics.to_csv(f"topics_name/topics_name_{temperature}.csv")
    print(topics)
