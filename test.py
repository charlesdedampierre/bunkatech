import pandas as pd
from bunkatech.topic_modeling.nested_topics import NestedTopicModeling

from sentence_transformers import SentenceTransformer


model = SentenceTransformer("/Users/charlesdedampierre/Desktop/all-MiniLM-L6-v2")
docs = ["this is a test"]
print("Territory embedding..")
emb = model.encode(docs, show_progress_bar=True)

"""data = pd.read_csv('/Volumes/OutFriend/shaping_ai/data_minus_neutre.csv')
data = data.sample(3000, random_state = 42)
# All the basic components to compute before more specialized computation

nested = NestedTopicModeling(data = data,
                        text_var = 'content',
                        index_var = 'unique_id',
                        extract_terms=True,
                        terms_embedding=True,
                        docs_embedding=True,
                        sample_size_terms=500,
                        terms_limit=500,
                        terms_ents=True,
                        terms_ngrams=(1, 2),
                        terms_ncs=True,
                        terms_include_pos=["NOUN", "PROPN", "ADJ"],
                        terms_include_types=["PERSON", "ORG"],
                        terms_embedding_model="distiluse-base-multilingual-cased-v1",
                        docs_embedding_model="tfidf",
                        language="en",
                        terms_path=None,
                        terms_embeddings_path=None,
                        docs_embeddings_path=None)

nested.fit(folding = None)

fig = nested.nested_maps(
        size_rule="equal_size",
        map_type="treemap",
        width=1000,
        height=1000,
        query=None)

"""

"""docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
df = pd.DataFrame(docs)
sample_size = 2000
df = df.sample(sample_size).reset_index(drop=True)
df.columns = ["text"]
df["bindex"] = df.index"""

# df.to_csv("sample_test.csv", index=False)
# df = pd.read_csv("sample_test.csv")


"""res = sbert_extension(
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

data = pd.read_csv(
    "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
    index_col=[0],
)
"""
