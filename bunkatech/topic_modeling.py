import pandas as pd
import warnings
from sklearn.cluster import KMeans
from .semantics.extract_terms import extract_terms_df
from .semantics.indexer import indexer
from .semantics.get_embeddings import get_embeddings
import umap
import plotly.express as px
from .visualization.make_bubble import wrap_by_word

# from .visualization.make_bubble import make_bubble
from .specificity import specificity

# from sklearn.cluster import OPTICS

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


class topic_modeling:
    def __init__(self) -> None:
        pass

    def fit(
        self,
        df: pd.DataFrame,
        index_var: str,
        text_var: str,
        sample_size: int = 2000,
        topic_number=20,
        max_terms=1000,
        top_terms=20,
        ngrams=(1, 2),
        ents=False,
        ncs=False,
        sample_terms=2000,
        model="tfidf",
        language="en",
    ):
        df = df.sample(min(len(df), sample_size))
        df = df[df[text_var].notna()]  # cautious not to let any nan
        df = df.reset_index(drop=True)

        print("Create Embeddings & Umap...")
        # Embeddings & UMAP
        embeddings_reduced, _ = get_embeddings(
            df,
            field=text_var,
            model=model,
            model_path="distiluse-base-multilingual-cased-v1",  # workds if sbert is chosen
        )

        df_emb = pd.DataFrame(embeddings_reduced)
        df_emb[index_var] = df[index_var]

        # clusters
        print("Create Clusters...")

        """df["cluster"] = (
            OPTICS(min_cluster_size=1 / 20).fit(embeddings_reduced).labels_.astype(str)
        )"""

        df["cluster"] = (
            KMeans(n_clusters=topic_number).fit(embeddings_reduced).labels_.astype(str)
        )

        topic_size = (
            df[["cluster", "bindex"]].groupby("cluster")["bindex"].count().reset_index()
        )
        topic_size.columns = ["cluster", "topic_size"]

        # Extract Terms
        terms = extract_terms_df(
            df,
            text_var=text_var,
            limit=max_terms,
            sample_size=sample_terms,
            ngs=True,  # ngrams
            ents=ents,  # entities
            ncs=ncs,  # nouns
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

        df_indexed = indexer(df[text_var].tolist(), list_terms, db_path=".")
        # get the Main form and the lemma
        df_indexed_full = pd.merge(
            df_indexed, df_terms, left_on="words", right_on="text"
        )
        df_indexed_full = df_indexed_full[["docs", "lemma", "main form", "text"]].copy()
        data = pd.merge(df_indexed_full, df, left_on="docs", right_on=text_var)
        data = data.drop("docs", axis=1)

        _, _, edge = specificity(
            data, X="cluster", Y="main form", Z=None, top_n=top_terms
        )

        topics = (
            edge.groupby("cluster")["main form"]
            .apply(lambda x: " | ".join(x))
            .reset_index()
        )
        topics = topics.rename(columns={"main form": "cluster_name"})
        topics = pd.merge(topics, topic_size, on="cluster")
        topics = topics.sort_values("topic_size", ascending=False)
        topics = topics.reset_index(drop=True)

        df_topics_names = pd.merge(df, topics, on="cluster")

        # only the vars
        self.df = df
        self.index_var = index_var
        self.text_var = text_var

        # Computes terms & topics list
        self.terms = terms
        self.topics = topics

        # With index
        self.df_topics_names = df_topics_names
        self.df_embeddings = df_emb
        self.df_indexed = data

        return self

    def visualize_embeddings(self, width: int = 1000, height: int = 1000):

        """Visualize the embeddings in 2D based on nested level.
        There is an hover for the text and clusters have names.

        """

        res = pd.merge(
            self.df_embeddings,
            self.df_topics_names,
            on=self.index_var,
        )
        res = pd.merge(
            res.drop([self.text_var, "cluster"], axis=1), self.df, on=self.index_var
        )

        if not hasattr(self, "embeddings_2d"):
            self.embeddings_2d = umap.UMAP(n_components=2).fit_transform(
                res[[0, 1, 2, 3, 4]]
            )

        res["dim_1"] = self.embeddings_2d[:, 0]
        res["dim_2"] = self.embeddings_2d[:, 1]

        res[self.text_var] = res[self.text_var].apply(lambda x: wrap_by_word(x, 10))
        res["cluster_label"] = (
            res["cluster"].astype(object) + " - " + res["cluster_name"]
        )
        res = res.dropna().reset_index(drop=True)

        fig = px.scatter(
            res,
            x="dim_1",
            y="dim_2",
            color="cluster_label",
            hover_data=[self.text_var],
            width=width,
            height=height,
        )

        return fig
