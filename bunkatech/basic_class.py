import pandas as pd
from .semantics.extract_terms import extract_terms_df
from .semantics.indexer import indexer
from sentence_transformers import SentenceTransformer


class BasicSemantics:
    """This Class carries out the basic operations that all following models will use
    - terms extraction
    - terms indexation
    - docs embeddings
    - terms embeddings

    This class will be used as Parent of other specialized class

    """

    def __init__(self) -> None:
        pass

    def fit(self, data, text_var, index_var):
        data = data[data[text_var].notna()].reset_index(drop=True)
        self.data = data
        self.text_var = text_var
        self.index_var = index_var

    def extract_terms(
        self,
        sample_size,
        limit,
        ents=True,
        ncs=True,
        ngrams=(1, 2),
        include_pos=["NOUN", "PROPN", "ADJ"],
        include_types=["PERSON", "ORG"],
        language="en",
        db_path=".",
    ):
        terms = extract_terms_df(
            self.data,
            text_var=self.text_var,
            limit=limit,
            sample_size=sample_size,
            ngs=True,  # ngrams
            ents=ents,  # entities
            ncs=ncs,  # nouns
            drop_emoji=True,
            remove_punctuation=False,
            ngrams=ngrams,
            include_pos=include_pos,
            include_types=include_types,
            language=language,
        )

        terms["main form"] = terms["main form"].apply(lambda x: x.lower())
        self.terms = terms

        self.index_terms(projection=False, db_path=db_path)

        return terms

    def terms_embeddings(self, embedding_model="distiluse-base-multilingual-cased-v1"):

        # Embed the terms with sbert
        model = SentenceTransformer(embedding_model)
        docs = list(self.terms["main form"])
        terms_embeddings = model.encode(docs, show_progress_bar=True)

        # Lower all the terms
        docs_prepro = [x.lower() for x in docs]

        # Create DataFrame
        df_embeddings = pd.DataFrame(terms_embeddings)
        df_embeddings.index = docs_prepro
        self.terms_embeddings = df_embeddings
        self.terms_embeddings_model = model

        return self.terms_embeddings

    def embeddings(self, embedding_model="distiluse-base-multilingual-cased-v1"):

        # Embed the docs with sbert
        model = SentenceTransformer(embedding_model)
        docs = list(self.data[self.text_var])
        docs_embeddings = model.encode(docs, show_progress_bar=True)

        # Create the Dataframe
        df_embeddings = pd.DataFrame(docs_embeddings)
        df_embeddings.index = self.data[self.index_var]
        # df_embeddings[self.index_var] = self.data[self.index_var]
        # df_embeddings.insert(0, self.index_var, df_embeddings.pop(self.index_var))

        self.embedding_model = embedding_model
        self.docs_embeddings = df_embeddings

        return self.docs_embeddings

    def index_terms(self, db_path=".", projection=False):

        """Intex the terms on the text_var dataset"""

        # Get all the differents types of terms (not only the main form)
        df_terms = self.terms.copy()
        df_terms["text"] = df_terms["text"].apply(lambda x: x.split(" | "))
        df_terms = df_terms.explode("text").reset_index(drop=True)

        # If new words from projection must be added
        if projection == True:
            list_terms = df_terms["text"].tolist() + self.projection_all
        else:
            list_terms = df_terms["text"].tolist()

        # Index the extracted terms
        df_indexed = indexer(
            self.data[self.text_var].tolist(), list_terms, db_path=db_path
        )

        # Merge the indexed terms with the df_terms
        df_indexed_full = pd.merge(
            df_indexed, df_terms, left_on="words", right_on="text"
        )

        # Merge with the initial dataset
        df_indexed_full = df_indexed_full[["docs", "lemma", "main form", "text"]].copy()
        df_enrich = pd.merge(
            self.data[[self.text_var, self.index_var]],
            df_indexed_full,
            left_on=self.text_var,
            right_on="docs",
        )
        df_enrich = df_enrich.drop("docs", axis=1)
        self.df_indexed = df_enrich

        return self
