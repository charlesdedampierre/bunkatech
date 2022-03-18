import pandas as pd
from pydantic import NoneStrBytes
from bunkatech.semantics.origami import Origami
from bunkatech.topic_modeling.nested_topics import NestedTopicModeling
from bunkatech.topic_modeling.topics import TopicModeling
from bunkatech.networks.networks_class import SemanticNetworks
from bunkatech.time.time import SemanticsTrends


class Bunka(
    NestedTopicModeling, Origami, SemanticsTrends, SemanticNetworks, TopicModeling
):
    def __init__(
        self,
        data,
        text_var,
        index_var,
        extract_terms=True,
        terms_embedding=False,
        docs_embedding=False,
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
    ):

        super().__init__(
            data,
            text_var,
            index_var,
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
            terms_path=terms_path,
            terms_embeddings_path=terms_embeddings_path,
            docs_embeddings_path=docs_embeddings_path,
        )

    def fit(self, date_var, popularity_var=None, folding=None):
        NestedTopicModeling.fit(self, folding=folding)
        if date_var is not None:
            SemanticsTrends.fit(self, date_var=date_var)
        if popularity_var is not None:
            self.popularity_var = popularity_var

    # Bunka manages the change of variables and especially the nestedness
    def change_variables(
        self,
        data=None,
        docs_embeddings=False,
    ):
        # Change the original variables
        if data is not None:
            self.data = data
            # If new data is added, then, the terms need to be re-indexed (or filteresd)
            self.index_terms(projection=False)

        if docs_embeddings:
            self.docs_embeddings_function(
                docs_embedding_model=self.docs_embedding_model
            )

    def get_top_popular_docs(self, top_n=10):
        top_docs = self.data[[self.index_var, self.text_var, self.popularity_var]]
        top_docs = top_docs.dropna()
        top_docs = top_docs.sort_values(self.popularity_var, ascending=False)
        top_docs = top_docs.head(top_n)
        self.top_docs = top_docs.set_index(self.index_var)
        return self.top_docs

    """def filter_nested_cluster(self, lv_0=None, lv_1=None, lv_2=None):

        if lv_0 == None and lv_1 == None and lv_2 == None:
            data_filtered = self.h_clusters_names.copy()

        elif lv_1 == None and lv_2 == None:
            mask = self.h_clusters_names["lemma_0"] == lv_0
            data_filtered = self.h_clusters_names[mask]

        elif lv_2 == None:
            mask = self.h_clusters_names["lemma_1"] == lv_1
            data_filtered = self.h_clusters_names[mask]

        else:
            mask = self.h_clusters_names["lemma_2"] == lv_2
            data_filtered = self.h_clusters_names[mask]

        self.data = pd.merge(
            data_filtered[[self.index_var]], self.data, on=self.index_var
        )

        return self.data

    def cluster_selection(df_clusters_name):

        level_0_list = list(set(df_clusters_name["lemma_0"].to_list())) + ["None"]

        default_ix = level_0_list.index("None")
        lv_0 = st.sidebar.selectbox(
            "Chose the Level 1", options=level_0_list, index=default_ix
        )

        # If None, then impossible to chose sub-topic
        if lv_0 == "None":
            level_1_list = ["None"]
        else:
            mask_1 = df_clusters_name["lemma_0"] == lv_0
            level_1_list = list(set(df_clusters_name[mask_1]["lemma_1"].to_list())) + [
                "None"
            ]

        default_ix = level_1_list.index("None")
        lv_1 = st.sidebar.selectbox(
            "Chose the Level 2", options=level_1_list, index=default_ix
        )

        # If None, then impossible to chose sub-topic in the selectbox
        if lv_1 == "None" or lv_0 == "None":
            level_2_list = ["None"]
        else:
            mask_2 = df_clusters_name["lemma_1"] == lv_1
            level_2_list = list(set(df_clusters_name[mask_2]["lemma_2"].to_list())) + [
                "None"
            ]

        default_ix = level_2_list.index("None")
        lv_2 = st.sidebar.selectbox(
            "Chose the Level 3", options=level_2_list, index=default_ix
        )

        # Data Selection based on choice

        if lv_0 == "None" and lv_1 == "None" and lv_2 == "None":
            data_filtered = df_clusters_name.copy()

        elif lv_1 == "None" and lv_2 == "None":
            mask = df_clusters_name["lemma_0"] == lv_0
            data_filtered = df_clusters_name[mask]

        elif lv_2 == "None":
            mask = df_clusters_name["lemma_1"] == lv_1
            data_filtered = df_clusters_name[mask]

        else:
            mask = df_clusters_name["lemma_2"] == lv_2
            data_filtered = df_clusters_name[mask]

        return data_filtered, lv_0, lv_1, lv_2

        data_filtered = cluster_selection(df_clusters_name)
            st.dataframe(data_filtered)
            st.write(len(data_filtered))

    """
