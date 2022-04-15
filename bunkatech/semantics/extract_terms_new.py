import pandas as pd
from tqdm import tqdm
import pickle


tqdm.pandas()
from multiprocessing import Pool
import multiprocessing

import textacy
import textacy.preprocessing
import textacy.representations
import textacy.tm

from functools import partial
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

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


def from_dict_to_frame(indexed_dict):
    data = {k: [v] for k, v in indexed_dict.items()}
    df = pd.DataFrame.from_dict(data).T
    df.columns = ["text"]
    df = df.explode("text")
    return df


def extract_terms(
    tuple,  # (index, text)
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

    index = tuple[0]
    text = tuple[1]

    prepro_text = preproc(str(text))
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

    doc = textacy.make_spacy_doc(prepro_text, lang=lang)

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
    multiprocess=True,
    sample_size=100000,
    drop_emoji=True,
    ngrams=(2, 2),
    remove_punctuation=True,
    include_pos=["NOUN", "PROPN", "ADJ"],
    include_types=["PERSON", "ORG"],
    language="en",
):

    data = data[data[text_var].notna()]
    data = data.sample(min(sample_size, len(data)))

    sentences = data[text_var].to_list()
    indexes = data.index.to_list()
    inputs = [(x, y) for x, y in zip(indexes, sentences)]

    if multiprocess is True:
        with Pool(multiprocessing.cpu_count() - 2) as p:
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

    else:
        res = list(
            tqdm(
                map(
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

    terms = (
        final_res.groupby(["text", "lemma", "ent", "ngrams"])
        .agg(count_terms=("text_index", "count"))
        .reset_index()
    )

    # duplicates to get rid of
    terms = terms.sort_values(["text", "ent"]).reset_index()
    terms = terms.drop_duplicates(["text"], keep="first")
    terms = terms.sort_values("count_terms", ascending=False)
    terms = terms.set_index("text")

    terms_indexed = (
        final_res[["text", "text_index"]].drop_duplicates().set_index("text")
    )
    terms_indexed = terms_indexed.rename(columns={"text_index": index_var})

    indexed_dict = (
        terms_indexed.reset_index().groupby(index_var)["text"].apply(list).to_dict()
    )

    return terms, indexed_dict


if __name__ == "__main__":
    data = pd.read_csv(
        "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/SHAI-CORPUS-MODEL-ALL-R2.csv",
        sep=";",
    )
    data = data.set_index("unique_id")

    terms, terms_indexed = extract_terms_df(
        data,
        text_var="title_clean",
        index_var="unique_id",
        ngs=True,
        ents=True,
        ncs=False,
        multiprocess=True,
        sample_size=100000,
        drop_emoji=True,
        ngrams=(1, 2),
        remove_punctuation=False,
        include_pos=["NOUN", "PROPN", "ADJ", "VERB"],
        include_types=["PER", "ORG", "LOC"],
        language="fr",
    )

    path = (
        "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/title"
    )
    terms.to_csv(path + "/terms.csv")

    with open(path + "/terms_indexed.pickle", "wb") as handle:
        pickle.dump(terms_indexed, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # terms_indexed.to_csv(path + "/terms_indexed.csv")"""
