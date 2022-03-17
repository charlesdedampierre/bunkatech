import pandas as pd
import plotly.express as px
import numpy as np
import warnings
import umap
from ..basic_class import BasicSemantics
from ..visualization.make_bubble import wrap_by_word

from ..networks.centroids import find_centroids
from sklearn.cluster import KMeans
from ..specificity import specificity
from .time_utils import cosine_distance_exponential_time_decay
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s : %(message)s", level=logging.INFO
)

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


class TopicModeling(BasicSemantics):
    def __init__(
        self,
        data,
        text_var,
        index_var,
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
        docs_embeddings_path=None,
    ) -> None:

        BasicSemantics.__init__(
            self,
            data=data,
            text_var=text_var,
            index_var=index_var,
            terms_path=terms_path,
            terms_embeddings_path=terms_embeddings_path,
            docs_embeddings_path=docs_embeddings_path,
        )

        BasicSemantics.fit(
            self,
            extract_terms=extract_terms,
            terms_embedding=terms_embedding,
            docs_embedding=docs_embedding,
            sample_size_terms=sample_size_terms,
            terms_limit=terms_limit,
            terms_ents=terms_ents,
            terms_ngrams=terms_ngrams,
            terms_ncs=terms_ncs,
            terms_include_pos=terms_include_pos,
            terms_include_types=terms_include_types,
            terms_embedding_model=terms_embedding_model,
            docs_embedding_model=docs_embedding_model,
            language=language,
        )
        """if self.date_var is not None:

            logging.info("Time Normalization...")

            self.data = self.data[self.data[self.date_var].notna()]
            self.data[self.date_var] = pd.to_datetime(self.data[self.date_var])
            timestamps = self.data[self.date_var].values.astype(np.int64) // 10 ** 9
            self.docs_embeddings = cosine_distance_exponential_time_decay(
                self.docs_embeddings,
                timestamps=timestamps,
                temperature=10,
            )"""

    def get_clusters(self, topic_number=20, top_terms=10):

        self.data["cluster"] = (
            KMeans(n_clusters=topic_number)
            .fit(self.docs_embeddings)
            .labels_.astype(str)
        )

        # Get the Topics Names
        df_clusters = pd.merge(
            self.data[[self.index_var, "cluster"]],
            self.df_indexed.reset_index(),
            on=self.index_var,
        )

        _, _, edge = specificity(
            df_clusters, X="cluster", Y="main form", Z=None, top_n=top_terms
        )

        topics = (
            edge.groupby("cluster")["main form"]
            .apply(lambda x: " | ".join(x))
            .reset_index()
        )
        topics = topics.rename(columns={"main form": "cluster_name"})

        # Get the Topics Size

        topic_size = (
            self.data[["cluster", self.index_var]]
            .groupby("cluster")[self.index_var]
            .count()
            .reset_index()
        )
        topic_size.columns = ["cluster", "topic_size"]

        topics = pd.merge(topics, topic_size, on="cluster")
        topics = topics.sort_values("topic_size", ascending=False)
        self.topics = topics.reset_index(drop=True)

        self.df_topics_names = pd.merge(
            self.data[[self.index_var, "cluster"]], topics, on="cluster"
        )

        return self.topics

    def visualize_embeddings(self, width: int = 1000, height: int = 1000):

        """
        Visualize the embeddings in 2D.
        There is an hover for the text and clusters have names.

        """

        res = pd.merge(
            self.docs_embeddings.reset_index(),
            self.df_topics_names,
            on=self.index_var,
        )
        res = pd.merge(res.drop("cluster", axis=1), self.data, on=self.index_var)

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

    def get_centroid_documents(self, top_elements: int = 2) -> pd.DataFrame:
        """Get the centroid documents of the clusters

        Returns
        -------
        pd.DataFrame
            the centroid_docs are separeated by ' || '

        """

        df_centroid = pd.merge(
            self.docs_embeddings.reset_index(),
            self.df_topics_names,
            on=self.index_var,
        )
        df_centroid = pd.merge(
            df_centroid.drop([self.text_var, "cluster"], axis=1),
            self.data,
            on=self.index_var,
        )

        df_centroid = df_centroid.rename(
            columns={0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
        )

        res = find_centroids(
            df_centroid,
            text_var=self.text_var,
            cluster_var="cluster",
            top_elements=top_elements,
            dim_lenght=5,
        )

        return res

    def temporal_topics(self):
        if self.date_var is None:
            raise ValueError("Please fit the class with the 'date' variable")
        return None
