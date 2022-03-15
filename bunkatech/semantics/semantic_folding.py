import pandas as pd
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly
import plotly.express as px
from sklearn.cluster import KMeans

# from utils import wrap_by_word
import numpy as np
from sentence_transformers import SentenceTransformer
from bunkatech.search.fts5_search import fts5_search


def wrap_by_word(string, n_words):
    """returns a string where \\n is inserted between every n words"""
    a = string.split()
    ret = ""
    for i in range(0, len(a), n_words):
        ret += " ".join(a[i : i + n_words]) + "<br>"

    return ret


def semantic_folding(
    semantic_group,
    data: pd.DataFrame,
    text_var: str,
    model: str = "tfidf",
    dimension_folding: int = 5,
    folding_only=False,
):

    docs = list(data[text_var])
    # automatic vocabulary extension

    full_research = pd.DataFrame()
    count = 1
    for query in semantic_group:
        print(query)
        res_search = fts5_search(query, docs)
        res_search["semantics"] = count
        # for the moment only give 1 if there is at least one term in the text
        res_search = res_search.groupby("docs")["semantics"].count().reset_index()
        res_search["semantics"] = count
        full_research = full_research.append(res_search)
        count += 1

    # Merge the result of the search with the initial dataset

    df_search = pd.merge(
        data, full_research, left_on=text_var, right_on="docs", how="left"
    )
    df_search = df_search.drop("docs", axis=1)
    # the 0 with be the category of everything that is no taken into account by the foldings

    # the folding_only paramter only keeps the data that stricly contains one of the terms
    if folding_only is True:
        df_search = df_search.dropna()
    else:
        df_search = df_search.fillna(0)

    # Create the model

    if model == "tfidf":
        model = TfidfVectorizer(max_features=20000)
        sentences = list(df_search[text_var])
        embeddings = model.fit_transform(sentences)

    elif model == "sbert":
        model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        docs = list(df_search[text_var])
        embeddings = model.encode(docs, show_progress_bar=True)

    # Realisze the semantic folding
    folding = umap.UMAP(
        n_components=dimension_folding, n_neighbors=10, metric="cosine", verbose=True
    )
    y = df_search["semantics"].to_list()

    # Feet the embeddings to the folding task
    folding.fit(embeddings, y)
    embeddings_folding = folding.transform(embeddings)

    return embeddings_folding


def visualising_folding(
    embeddings_folding: np.array, data: pd.DataFrame, text_var: str, n_clusters: int = 3
):

    # Vosualize the results in 2 dimentions
    df_emb = pd.DataFrame(embeddings_folding, columns=["dim_1", "dim_2"])

    df_emb["clusters"] = (
        KMeans(n_clusters=n_clusters).fit(embeddings_folding).labels_.astype(str)
    )
    df_emb[text_var] = data[text_var].apply(lambda x: wrap_by_word(x, 10))

    fig = px.scatter(
        df_emb,
        x="dim_1",
        y="dim_2",
        color="clusters",
        hover_data=[text_var],
        width=1500,
        height=1500,
    )
    return fig


if __name__ == "__main__":

    folding_1 = ["problèmes", "danger", "menace"]
    folding_2 = ["bienfaits", "opportunité", "génial", "innovation"]
    semantic_group = [folding_1, folding_2]

    data = pd.read_excel(
        "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/labeling/SHAI-LABELS-ROUND-1.xlsx"
    )

    embeddings = semantic_folding(
        semantic_group, data, text_var="title_lead", model="tfidf", dimension_folding=2
    )

    fig = visualising_folding(
        embeddings_folding=embeddings, data=data, text_var="title_lead", n_clusters=3
    )
    fig.show()
