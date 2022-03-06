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

df = pd.read_csv("sample_test.csv")

bunka = NestedTopicModeling()
model = bunka.fit(
    df,
    text_var="text",
    index_var="bindex",
    sample_size=500,
    sample_terms=1000,
    embeddings_model="tfidf",
    ngrams=(1, 2),
    ents=False,
    language="en",
    db_path=".",
)

map = bunka.make_nested_maps(
    size_rule="topic_documents", map_type="treemap", query="president"
)
map.show()
