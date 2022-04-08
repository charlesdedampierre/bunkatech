import pandas as pd
from fuzzywuzzy import fuzz, process
from tqdm import tqdm
import numpy as np

tqdm.pandas()
from multiprocessing import Pool

multiprocessing_pools = 8

import textacy
import textacy.preprocessing
import textacy.representations
import textacy.tm

from functools import partial
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
import bamboolib

from sentence_transformers import SentenceTransformer

preproc = textacy.preprocessing.make_pipeline(
    textacy.preprocessing.normalize.unicode,
    textacy.preprocessing.normalize.bullet_points,
    textacy.preprocessing.normalize.quotation_marks,
    textacy.preprocessing.normalize.whitespace,
    textacy.preprocessing.normalize.hyphenated_words,
    textacy.preprocessing.remove.brackets,
    textacy.preprocessing.replace.currency_symbols,
    textacy.preprocessing.remove.html_tags,
)


def extract_terms(
    tuple,
    ngs=True,
    ents=True,
    ncs=False,
    ngrams=(2, 2),
    drop_emoji=True,
    remove_punctuation=False,
    include_pos=["NOUN", "PROPN", "ADJ"],
    include_types=["PERSON", "ORG"],
    language="en",
):

    index = tuple(1)
    text = tuple(0)

    prepro_text = preproc(text)
    if drop_emoji == True:
        prepro_text = textacy.preprocessing.replace.emojis(prepro_text, repl="")

    if remove_punctuation == True:
        prepro_text = textacy.preprocessing.remove.punctuation(prepro_text)

    if language == "zh":
        lang = textacy.load_spacy_lang("zh_core_web_sm", disable=())
    if language == "en":
        lang = textacy.load_spacy_lang("en_core_web_sm", disable=())
    elif language == "fr":
        lang = textacy.load_spacy_lang("fr_core_news_lg", disable=())

    doc = textacy.make_spacy_doc(text, lang=lang)

    terms = []

    if ngs:
        ngrams_terms = list(
            textacy.extract.terms(
                doc,
                ngs=partial(
                    textacy.extract.ngrams,
                    n=ngrams,
                    filter_punct=True,
                    filter_stops=True,
                    include_pos=include_pos,
                ),
                dedupe=False,
            )
        )

        terms.append(ngrams_terms)

    if ents:
        ents_terms = list(
            textacy.extract.terms(
                doc,
                ents=partial(textacy.extract.entities, include_types=include_types),
                dedupe=False,
            )
        )
        terms.append(ents_terms)

    if ncs:
        ncs_terms = list(
            textacy.extract.terms(
                doc,
                ncs=partial(textacy.extract.noun_chunks, drop_determiners=True),
                dedupe=False,
            )
        )

        noun_chunks = [x for x in ncs_terms if len(x) >= 3]
        terms.append(noun_chunks)

    final = [item for sublist in terms for item in sublist]
    final = list(set(final))

    df = [
        (term.text, term.lemma_.lower(), term.label_, term.__len__()) for term in final
    ]
    df = pd.DataFrame(df, columns=["text", "lemma", "ent", "ngrams"])
    df["text_index"] = index

    return df


def extract_terms_df(
    data,
    text_var,
    index_var,
    ngs=True,
    ents=True,
    ncs=False,
    multiprocessing=True,
    sample_size=2000,
    drop_emoji=True,
    ngrams=(2, 2),
    remove_punctuation=True,
    include_pos=["NOUN", "PROPN", "ADJ"],
    include_types=["PERSON", "ORG"],
    language="en",
):

    data = data[data[text_var].notna()]
    data = data.sample(min(sample_size, len(data)))

    sentences = data["text"].to_list()
    indexes = data.index.to_list()
    inputs = [(x, y) for x, y in zip(indexes, sentences)]

    if multiprocessing is True:
        with Pool(multiprocessing_pools) as p:
            res = list(
                tqdm(
                    p.imap(
                        partial(
                            extract_terms,
                            ngs=ngs,
                            ents=ents,
                            ncs=ncs,
                            drop_emoji=drop_emoji,
                            remove_punctuation=remove_punctuation,
                            ngrams=ngrams,
                            include_pos=include_pos,
                            include_types=include_types,
                            language=language,
                        ),
                        inputs,
                    ),
                    total=len(inputs),
                )
            )

        final_res = pd.concat([x for x in res])

    """res = pd.DataFrame()
    for x in tqdm(range(len(data)), total=len(data)):

        text = data.iloc[x][text_var]
        index = data.iloc[x][index_var]

        df = extract_terms(index=index, text=text, ngs=ngs, ents=ents, ncs=ncs)
        res = res.append(df)
    """

    func = lambda x: " | ".join(x)

    terms = (
        final_res.groupby(["text", "lemma", "ent", "ngrams"])
        .agg(count_terms=("text_index", "count"))
        .reset_index()
    )

    terms = terms.set_index("text")

    df_index = final_res[["text", index_var]].drop_duplicates().set_index("text")

    return terms, df_index
