from bunkatech.nested_topic_modeling import NestedTopicModeling
from bunkatech.semantics.vocabulary_extension import sbert_extension
import pandas as pd
<<<<<<< HEAD
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
df = pd.DataFrame(docs)
sample_size = 2000
df = df.sample(sample_size).reset_index(drop=True)
df.columns = ["text"]
df["bindex"] = df.index

#df.to_csv("sample_test.csv", index=False)
#df = pd.read_csv("sample_test.csv")

=======

from bunkatech.semantics.origami import Origami
from bunkatech.time import SemanticsTrend
from bunkatech import BasicSemantics

from bunkatech.topic_modeling.nested_topics import NestedTopicModeling

"""data = pd.read_csv(
    "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
    index_col=[0],
)
"""
data = pd.read_excel(
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/labeling/SHAI-LABELS-ROUND-1.xlsx"
)

data["bindex"] = data.index

nested = NestedTopicModeling(data=data, text_var="title_lead", index_var="bindex")
folding_1 = ["problèmes", "danger", "menace"]
folding_2 = ["bienfaits", "opportunité", "génial", "innovation"]
folding = [folding_1, folding_2]

nested.fit(
    folding=folding,
    docs_embedding_model="tfidf",
    extract_terms=True,
    terms_embedding=False,
    sample_size_terms=500,
    terms_limit=500,
    terms_ents=False,
    terms_ngrams=(1, 2),
    terms_ncs=False,
    terms_include_pos=["NOUN", "PROPN", "ADJ"],
    terms_include_types=["PERSON", "ORG"],
    terms_embedding_model="distiluse-base-multilingual-cased-v1",
    language="fr",
)

fig = nested.visualize_embeddings()
fig.show()


# h_clusters_names = nested.nested_frame()
# print(h_clusters_names)

"""class NewClass(Bourdieu, SemanticsTrend):
    def __init__(self, data, text_var, index_var, date_var) -> None:
        SemanticsTrend.__init__(self, data, text_var, index_var, date_var)

        SemanticsTrend.fit(
            self,
            extract_terms=True,
            docs_embedding=False,
            terms_embedding=True,
            sample_size_terms=100,
            terms_limit=100,
            terms_ents=False,
            terms_ngrams=(2, 2),
            terms_ncs=False,
            language="en",
        )

    def trend(self):
        fig = SemanticsTrend.moving_average_comparison(self)
        return fig

    def projection(self, projection_1, projection_2):
        fig = Bourdieu.bourdieu_projection(self, projection_1, projection_2)
        return fig


df_sample = pd.read_csv(
    "/Volumes/OutFriend/timeline_folding/time_sample.csv", index_col=[0]
)
platform = "facebook"
df_sample = df_sample[df_sample["origin"] == platform]
df_sample["date"] = pd.to_datetime(df_sample["date"])
df_sample = df_sample.rename(columns={"text": "fb_text"})


test = NewClass(data=df_sample, text_var="fb_text", index_var="id", date_var="date")
fig = test.projection(projection_1=["good", "bad"], projection_2=["vaccin", "naturel"])
fig.show()"""


'''# df = pd.read_csv("sample_test.csv")

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
>>>>>>> be660dbd9976d632d4e887ab203a9939274341bc

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
'''
