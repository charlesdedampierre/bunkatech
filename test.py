import pandas as pd
import numpy as np
import plotly


"""
from bunkatech.semantics.extract_terms import extract_terms_df
from bunkatech.semantics.indexer import indexer
from bunkatech.semantics.get_embeddings import get_embeddings
from bunkatech.nested_topic_modeling import nested_topic_modeling

from bunkatech.hierarchical_clusters import hierarchical_clusters
from bunkatech.hierarchical_clusters_label import hierarchical_clusters_label

from bunkatech.visualization.make_bubble import make_bubble
from bunkatech.visualization.sankey import make_sankey
from bunkatech.visualization.topics_treemap import topics_treemap
from bunkatech.visualization.topics_sunburst import topics_sunburst
from sklearn.datasets import fetch_20newsgroups
"""

from bunkatech.networks.centroids import find_centroids

data = pd.read_csv("centroids_test.csv")

res = find_centroids(
    data, text_var="text", cluster_var="clusters", top_elements=2, dim_lenght=5
)

print(res.to_csv("centered_documents.csv"))

"""docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
df = pd.DataFrame(docs)
df = df.sample(2000)
df = df.reset_index(drop=True)
df.columns = ["text"]
df["bindex"] = df.index


model = nested_topic_modeling()

bunka = model.fit(
    df,
    text_var="text",
    index_var="bindex",
    sample_size=5000,
    sample_terms=200,
    embeddings_model="tfidf",
    ngrams=(1, 2),
    ents=False,
    language="en",
    db_path=".",
)

fig = model.visualize_embeddings(n_clusters=20)
fig.show()
"""
