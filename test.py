from bunkatech.topic_modeling import topic_modeling
from bunkatech.nested_topic_modeling import NestedTopicModeling
from bunkatech.semantics.vocabulary_extension import sbert_extension
import pandas as pd

# df = pd.read_csv("sample_test.csv")

df = pd.read_excel(
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/labeling/SHAI-LABELS-ROUND-1.xlsx"
)

df = df.reset_index(drop=True)
df["bindex"] = df.index

folding = [
    ["bien", "opportunité", "innovation", "amélioration", "promesses"],
    ["danger", "tuer", "problèmes", "mauvais", "risques", "dangers", "peur"],
    ["savoir", "connaissance", "limite", "cerveau", "immense"],
]

res = sbert_extension(
    folding[0],
    df,
    text_var="title_lead",
    sample_terms=1000,
    language="fr",
    top_n=30,
    bert_model="/Volumes/OutFriend/sbert_model/distiluse-base-multilingual-cased-v1",
)

extented_folding = res["main form"].to_list()

print(extented_folding)

bunka = NestedTopicModeling()
model = bunka.fit(
    df,
    text_var="title_lead",
    index_var="bindex",
    sample_size=1000,
    sample_terms=500,
    embeddings_model="tfidf",
    folding=[extented_folding],  # there can be different categories of folding
    ngrams=(2, 2),
    ents=False,
    language="fr",
    db_path=".",
)

map = bunka.make_nested_maps(
    size_rule="topic_documents",
    map_type="treemap",
    query=extented_folding,
)
map.show()

"""
data = pd.read_csv(
    "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
    index_col=[0],
)
"""
