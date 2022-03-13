import pandas as pd
from gensim.models import Word2Vec as wv
from .utils import erase_comma, erase_point, erase_accent
from sentence_transformers import SentenceTransformer
from .extract_terms import extract_terms_df
from ..search.query import make_query


def sbert_extension(
    queries: list,
    data,
    text_var,
    sample_terms=3000,
    language="en",
    top_n: int = 20,
    bert_model='"distiluse-base-multilingual-cased-v1"',
):
    """Extract Similar terms based on a query and sbert and output the top_n most similar terms"""
    data = data[data[text_var].notna()]
    terms = extract_terms_df(
        data,
        text_var=text_var,
        limit=4000,
        sample_size=sample_terms,
        ngs=True,  # ngrams
        ents=True,  # entities
        ncs=True,  # nouns
        drop_emoji=True,
        remove_punctuation=False,
        ngrams=(1, 2),
        include_pos=["NOUN", "PROPN", "ADJ"],
        include_types=["PERSON", "ORG"],
        language=language,
    )

    terms["bindex"] = terms.index
    # embed the temrs with sbert
    model = SentenceTransformer(bert_model)
    docs = list(terms["main form"])
    terms_embeddings = model.encode(docs, show_progress_bar=True)

    semantic_words = pd.DataFrame()
    for word in queries:

        try:
            res = make_query(
                data=terms,
                model=model,
                corpus_embeddings=terms_embeddings,
                query=[word],
                data_id="bindex",
                top_n=top_n,
            )

            fin = pd.merge(res, terms, on="bindex")
            fin = fin[["query", "main form"]]
            semantic_words = semantic_words.append(fin)
        except:
            pass

    semantic_words = semantic_words.reset_index(drop=True)
    return semantic_words


def word2vec_extension(
    data: pd.DataFrame, text_var: str = "title_lead", folding: list = ["voiture"]
) -> pd.DataFrame:
    """Create a Vocabulary extension fior a list of words based on word2vec Algorihtms"""

    docs = data[text_var].to_list()
    final = [doc.split() for doc in docs]
    model = wv(
        sentences=final, vector_size=100, window=10, min_count=3, workers=8, epochs=15
    )

    df_final = pd.DataFrame()

    for word in folding:
        try:
            res = model.wv.most_similar(word, topn=10)
            df = pd.DataFrame(res, columns=["similar_words", "score"])
            df["word"] = word
            df_final = df_final.append(df)
        except:
            pass

    df_final = df_final[["word", "similar_words", "score"]]
    df_final["similar_words"] = df_final["similar_words"].apply(lambda x: x.lower())
    df_final["similar_words"] = df_final["similar_words"].apply(
        lambda x: erase_comma(x)
    )
    df_final["similar_words"] = df_final["similar_words"].apply(
        lambda x: erase_point(x)
    )
    df_final["similar_words"] = df_final["similar_words"].apply(
        lambda x: erase_accent(x)
    )

    return df_final
