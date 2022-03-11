from bunkatech.topic_modeling import topic_modeling
from bunkatech.nested_topic_modeling import NestedTopicModeling

import pandas as pd

# df = pd.read_csv("sample_test.csv")

"""df = pd.read_excel(
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/labeling/SHAI-LABELS-ROUND-1.xlsx"
)

df = df.reset_index(drop=True)
df["bindex"] = df.index

folding = [
    ["bien", "avancé", "opportunité", "innovation", "amélioration", "promesses"],
    ["danger", "tuer", "problèmes", "mauvais", "risques", "dangers", "peur"],
    ["savoir", "connaissance", "limite", "cerveau", "immense"],
]


bunka = NestedTopicModeling()
model = bunka.fit(
    df,
    text_var="title_lead",
    index_var="bindex",
    sample_size=1000,
    sample_terms=500,
    embeddings_model="tfidf",
    folding=None,
    ngrams=(2, 2),
    ents=False,
    language="fr",
    db_path=".",
)

map = bunka.make_nested_maps(
    size_rule="topic_documents",
    map_type="treemap",
    query=folding[2],
)
map.show()
"""

from bunkatech.semantics.vocabulary_extension import sbert_extension

data = pd.read_csv(
    "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
    index_col=[0],
)


exploration_words = ["success"]


res = sbert_extension(
    exploration_words,
    data,
    text_var="description",
    sample_terms=1000,
    language="en",
    top_n=30,
)

print(res)
