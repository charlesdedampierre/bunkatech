import pandas as pd
from .semantics.extract_terms import extract_terms_df
from .semantics.indexer import indexer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import umap


class BasicSemantics:
    """This Class carries out the basic operations that all following models will use
    - terms extraction
    - terms indexation
    - docs embeddings
    - terms embeddings

    This class will be used as Parent of other specialized class

    """

    def __init__(
        self,
        data,
        text_var,
        index_var,
        terms_path=None,
        terms_embeddings_path=None,
        docs_embeddings_path=None,
    ):
        self.data = data[data[text_var].notna()].reset_index(drop=True)
        self.text_var = text_var
        self.index_var = index_var

        self.terms_path = terms_path
        self.terms_embeddings_path = terms_embeddings_path
        self.docs_embeddings_path = docs_embeddings_path

        # Load existing dataset if they exist
        if terms_path is not None:
            self.terms = pd.read_csv(terms_path, index_col=[0])
            self.index_terms(projection=False, db_path=".")

        if terms_embeddings_path is not None:
            self.terms_embeddings = pd.read_csv(terms_embeddings_path, index_col=[0])

        if docs_embeddings_path is not None:
            self.docs_embeddings = pd.read_csv(docs_embeddings_path, index_col=[0])

    def fit(
        self,
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
        docs_dimension_reduction=None,
        language="en",
    ):

        if self.terms_embeddings_path is not None:
            terms_embedding = False
        if self.docs_embeddings_path is not None:
            docs_embedding = False
        if self.terms_path is not None:
            extract_terms = False

        if extract_terms:
            self.terms = self.extract_terms(
                sample_size=sample_size_terms,
                limit=terms_limit,
                ents=terms_ents,
                ncs=terms_ncs,
                ngrams=terms_ngrams,
                include_pos=terms_include_pos,
                include_types=terms_include_types,
                language=language,
            )

        if terms_embedding:
            self.terms_embeddings = self.terms_embeddings_function(
                terms_embedding_model=terms_embedding_model
            )

        if docs_embedding:
            self.docs_embeddings = self.docs_embeddings_function(
                docs_embedding_model=docs_embedding_model,
                dimension_reduction=docs_dimension_reduction,
            )

        self.docs_embedding_model = docs_embedding_model
        self.terms_embedding_model = terms_embedding_model

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

    def terms_embeddings_function(
        self, terms_embedding_model="distiluse-base-multilingual-cased-v1"
    ):

        # Embed the terms with sbert
        model = SentenceTransformer(terms_embedding_model)
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

    def docs_embeddings_function(
        self,
        docs_embedding_model="distiluse-base-multilingual-cased-v1",
        dimension_reduction=None,
    ):

        if docs_embedding_model == "tfidf":
            model = TfidfVectorizer(max_features=3000)
            sentences = list(self.data[self.text_var])
            docs_embeddings = model.fit_transform(sentences)
            docs_embeddings = docs_embeddings.todense()

        else:
            # Embed the docs with sbert
            model = SentenceTransformer(docs_embedding_model)
            docs = list(self.data[self.text_var])
            docs_embeddings = model.encode(docs, show_progress_bar=True)

        if dimension_reduction is not None:
            docs_embeddings = umap.UMAP(
                n_components=dimension_reduction, verbose=True
            ).fit_transform(docs_embeddings)

        # Create the Dataframe
        df_embeddings = pd.DataFrame(docs_embeddings)
        df_embeddings.index = self.data[self.index_var]

        self.docs_embedding_model = docs_embedding_model
        self.docs_embeddings = df_embeddings

        return self.docs_embeddings

    def index_terms(self, db_path=".", projection=False):

        # Projection is used when one add new words to the terms

        """

        Index the terms on the text_var dataset

        """

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
        self.df_indexed = df_enrich.drop(["docs", self.text_var], axis=1)
        self.df_indexed = self.df_indexed.set_index(self.index_var)

        return self
