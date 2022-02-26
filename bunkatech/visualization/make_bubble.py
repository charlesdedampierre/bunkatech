import plotly.express as px
import umap
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly
from sklearn.cluster import KMeans

import numpy as np


def wrap_by_word(string, n_words):
    """returns a string where \\n is inserted between every n words"""
    try:
        a = string.split()
        ret = ""
        for i in range(0, len(a), n_words):
            ret += " ".join(a[i : i + n_words]) + "<br>"
    except:
        pass

    return ret


def make_bubble(emb, data, text_var, n_clusters=20, width=1000, height=1000):

    print("UMAP Reduction...")
    X_embedded_fit = umap.UMAP(n_components=2).fit_transform(emb)

    df_emb = pd.DataFrame(X_embedded_fit, columns=["dim_1", "dim_2"])

    df_emb["clusters"] = (
        KMeans(n_clusters=n_clusters).fit(X_embedded_fit).labels_.astype(str)
    )
    df_emb[text_var] = data[text_var].apply(lambda x: wrap_by_word(x, 10))

    fig = px.scatter(
        df_emb,
        x="dim_1",
        y="dim_2",
        color="clusters",
        hover_data=[text_var],
        width=width,
        height=height,
    )

    return fig
