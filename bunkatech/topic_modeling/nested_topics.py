import pandas as pd
import plotly.express as px
import warnings
import umap
from ..basic_class import BasicSemantics

from ..semantics.semantic_folding import semantic_folding

from .hierarchical_clusters import hierarchical_clusters
from .hierarchical_clusters_label import hierarchical_clusters_label

from ..visualization.make_bubble import wrap_by_word
from ..visualization.sankey import make_sankey
from ..visualization.topics_nested import topics_nested

from ..networks.centroids import find_centroids
from ..search.fts5_search import fts5_search


warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


class NestedTopicModeling(BasicSemantics):
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
        docs_dimension_reduction=5,
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
            docs_dimension_reduction=docs_dimension_reduction,
        )

    def fit(self, folding=None):
        # When fit, get the Folding and the Structure of the nested clusters

        if folding is not None:

            embeddings_folding = semantic_folding(
                folding,
                self.data,
                text_var=self.text_var,
                model=self.docs_embedding_model,
                dimension_folding=5,
                folding_only=False,
            )

            self.docs_embeddings = pd.DataFrame(embeddings_folding)
            self.docs_embeddings.index = self.data[self.index_var]

        #  Get the  specific computation
        self.nested_frame()

    def nested_frame(self):

        # Create clusters
        h_clusters = hierarchical_clusters(self.docs_embeddings.reset_index())
        merge = pd.merge(h_clusters, self.df_indexed.reset_index(), on=self.index_var)

        # Create clusters names
        self.h_clusters_names = hierarchical_clusters_label(merge)
        self.h_clusters_names = self.h_clusters_names.drop(
            ["main form", "text"], axis=1
        )
        self.h_clusters_names = self.h_clusters_names.drop_duplicates().reset_index(
            drop=True
        )
        self.h_clusters_names_initial = self.h_clusters_names.copy()

        return self.h_clusters_names

    def nested_maps(
        self,
        size_rule="equal_size",
        map_type="treemap",
        width=1000,
        height=1000,
        query=None,
    ):

        """Create Maps to display information

        Parameters
        ----------
        size_rule : str, optional
            _description_, by default "equal_size"
        map_type : str, optional
            _description_, by default "treemap"
        width : int, optional
            _description_, by default 1000
        height : int, optional
            _description_, by default 1000
        query : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        self.nested_frame()
        # Create the clusters and the names

        if query is not None:
            docs = self.data[self.text_var].to_list()
            # Make the term search among the documents
            res_search = fts5_search(query, docs)

            # Merge the results with the text_var
            df_cluster_names_filter = pd.merge(
                self.data, res_search, left_on=self.text_var, right_on="docs"
            )

            # Merge the ids with the nested clusters if information
            df_cluster_names_filter = pd.merge(
                self.h_clusters_names,
                df_cluster_names_filter[[self.index_var]],
                on=self.index_var,
            )

            # group by the more nested topics
            count_search = (
                df_cluster_names_filter.groupby(["lemma_2"])
                .agg(score_norm=(self.index_var, "count"))
                .reset_index()
            )
        else:
            count_search = None

        if map_type == "treemap":
            # Make treemap
            map = topics_nested(
                nested_topics=self.h_clusters_names,
                index_var=self.index_var,
                size_rule=size_rule,
                width=width,
                height=height,
                count_search=count_search,
                map_type="treemap",
            )

        elif map_type == "sunburst":
            # Make sunburst
            map = topics_nested(
                nested_topics=self.h_clusters_names,
                index_var=self.index_var,
                size_rule=size_rule,
                width=width,
                height=height,
                count_search=count_search,
                map_type="sunburst",
            )

        elif map_type == "icicle":
            # Make sunburst
            map = topics_nested(
                nested_topics=self.h_clusters_names,
                index_var=self.index_var,
                size_rule=size_rule,
                width=width,
                height=height,
                count_search=count_search,
                map_type="icicle",
            )

        elif map_type == "sankey":
            # Make Sankey Diagram
            map = make_sankey(
                self.h_clusters_names,
                field="Dataset",
                index_var=self.index_var,
                width=width,
                height=height,
            )

        else:
            raise ValueError(
                f' "{map_type}" is not a correct value. Please enter a correct map_type value such as "Treemap", "Sunburst" or "Sankey"'
            )

        return map

    def visualize_2D_embeddings(
        self, nested_level: int = 0, width: int = 1000, height: int = 1000
    ):

        """Visualize the embeddings in 2D based on nested level.
        There is an hover for the text and clusters have names.

        """

        res = pd.merge(
            self.docs_embeddings.reset_index(),
            self.h_clusters_names,
            on=self.index_var,
        )

        res = pd.merge(res, self.data, on=self.index_var)

        if not hasattr(self, "embeddings_2d"):
            self.embeddings_2d = umap.UMAP(n_components=2, verbose=True).fit_transform(
                res[[0, 1, 2, 3, 4]]
            )

        res["dim_1"] = self.embeddings_2d[:, 0]
        res["dim_2"] = self.embeddings_2d[:, 1]

        res[f"level_{nested_level}"] = res[f"level_{nested_level}"].astype(object)
        res[self.text_var] = res[self.text_var].apply(lambda x: wrap_by_word(x, 10))

        res["clusters"] = (
            res[f"level_{nested_level}"].astype(str)
            + " - "
            + res[f"lemma_{nested_level}"]
        )

        fig = px.scatter(
            res,
            x="dim_1",
            y="dim_2",
            color="clusters",
            hover_data=[self.text_var],
            width=width,
            height=height,
        )

        return fig

    def get_centroid_documents_nested(
        self, nested_level: int = 0, top_elements: int = 2
    ) -> pd.DataFrame:
        """For a given nestednness, Get the centroid element(s), the farest element
        and the cluster radius (farest element - centroid element)


        Returns
        -------
        pd.DataFrame
            the centroid_docs are separeated by ' || '

        """

        # Merge embeddings and text_var
        df_centroid = pd.merge(
            self.docs_embeddings.reset_index(), self.data, on=self.index_var
        )

        # merge embeddings & clusters names
        df_centroid = pd.merge(df_centroid, self.h_clusters_names, on=self.index_var)
        df_centroid = df_centroid.rename(
            columns={0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
        )

        res = find_centroids(
            df_centroid,
            text_var=self.text_var,
            cluster_var=f"level_{nested_level}",
            top_elements=top_elements,
            dim_lenght=5,
        )

        return res

    def temporal_topics_nested(
        self,
        date_var,
        width=1000,
        height=500,
        normalize_y=True,
        min_range=None,
        max_range=None,
    ):
        """Plot the topics evolution in time"""

        df_time = self.data[[self.index_var, date_var]].copy()
        df_time = pd.merge(df_time, self.h_clusters_names, on=self.index_var)
        df_time = (
            df_time.groupby([date_var, "lemma_0"])
            .agg(count_docs=(self.index_var, "count"))
            .reset_index()
        )
        df_time["count_docs_date"] = df_time.groupby([date_var])[
            "count_docs"
        ].transform(lambda x: sum(x))

        df_time["normalized_count_docs"] = (
            df_time["count_docs"] / df_time["count_docs_date"]
        )

        if min_range is not None and max_range is not None:
            range_x = (min_range, max_range)
        else:
            range_x = (min(df_time[date_var]), max(df_time[date_var]))

        if normalize_y is True:
            y = "normalized_count_docs"
        else:
            y = "count_docs"

        df_time = df_time.rename(columns={"lemma_0": "topics_names"})

        fig = px.bar(
            df_time,
            x=date_var,
            y=y,
            color="topics_names",
            width=width,
            height=height,
            range_x=range_x,
        )

        self.date_var = date_var
        self.df_time = df_time

        return fig
