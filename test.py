from bunkatech.topic_modeling import topic_modeling
from bunkatech.nested_topic_modeling import NestedTopicModeling

import pandas as pd

"""
docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
df = pd.DataFrame(docs)
sample_size = 200
df = df.sample(sample_size).reset_index(drop=True)
df.columns = ["text"]
df["bindex"] = df.index
df.to_csv("sample_test.csv", index=False)"""

# df = pd.read_csv("sample_test.csv")

df = pd.read_excel(
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
