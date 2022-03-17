import pandas as pd
from bunkatech.semantics.origami import Origami
from bunkatech.topic_modeling.nested_topics import NestedTopicModeling
from bunkatech.networks.network_class import SemanticNetwork
from bunkatech.basic_class import BasicSemantics
from bunkatech.time.time import SemanticsTrend


class Bunka(NestedTopicModeling):
    def __init__(self, data, text_var, index_var):

        NestedTopicModeling.__init__(self, data, text_var, index_var)

    def fit(
        self,
        folding=None,
        extract_terms=True,
        terms_embedding=True,
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
    ):

        NestedTopicModeling.fit(
            self,
            folding=folding,
            extract_terms=extract_terms,
            terms_embedding=terms_embedding,
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
