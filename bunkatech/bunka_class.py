import pandas as pd
from bunkatech.semantics.origami import Origami
from bunkatech.topic_modeling.nested_topics import NestedTopicModeling
from bunkatech.networks.networks_class import SemanticNetworks
from bunkatech.time.time import SemanticsTrends


class Bunka(NestedTopicModeling, Origami, SemanticsTrends, SemanticNetworks):
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

    def fit(self, date_var, folding=None):
        NestedTopicModeling.fit(self, folding=folding)
        SemanticsTrends.fit(self, date_var=date_var)
