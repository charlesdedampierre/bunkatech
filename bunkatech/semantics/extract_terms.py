import pandas as pd
import textacy
import textacy.preprocessing
import textacy.representations
import textacy.tm
from functools import partial
import numpy as np
from tqdm import tqdm


def chi2_table(doc_term_matrix: np.array, vocab: dict) -> pd.DataFrame:
    """Takes a doc_term_matrix, a dictionary of vocabulary and id and compute the chi2 score

    Args:
        doc_term_matrix (np.array): matrix with docs as rows and terms as columns
        vocab (dict):  dictionary of vocabulary and id

    Returns:
        pd.DataFrame with two colums
        terms
        chi2
    """

    # Add column name to the matrix
    matrix = pd.DataFrame(doc_term_matrix.todense())
    list_columns = list(matrix.columns)
    id_to_term = {v: k for k, v in vocab.items()}
    new_columns = [id_to_term[x] for x in list_columns]
    matrix.columns = new_columns

    # Average the chi2 score for every term
    df_chi2 = matrix.sum().reset_index()
    df_chi2.columns = ["lemma", "chi2"]
    df_chi2 = df_chi2.sort_values("chi2", ascending=False).reset_index(drop=True)

    return df_chi2


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
    text,
    ngs=True,
    ents=True,
    ncs=True,
    drop_emoji=True,
    remove_punctuation=False,
    ngrams=(2, 2),
    include_pos=["NOUN", "PROPN", "ADJ"],
    include_types=["PERSON", "ORG"],
    language="en",
):

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
    doc = textacy.make_spacy_doc(prepro_text, lang=lang)

    terms = []
    if ngs is True:
        ngrams_terms = list(
            textacy.extract.terms(
                doc,
                ngs=partial(
                    textacy.extract.ngrams,
                    n=ngrams,
                    include_pos=include_pos,
                    filter_punct=True,
                    filter_stops=True,
                ),
                dedupe=False,
            )
        )
        terms.append(ngrams_terms)

    if ents is True:
        ents_terms = list(
            textacy.extract.terms(
                doc,
                ents=partial(textacy.extract.entities, include_types=include_types),
                dedupe=False,
            )
        )
        terms.append(ents_terms)

    if ncs is True:
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
    # final = ngrams_terms + ents_terms + noun_chunks
    final = list(set(final))

    df = [(term.text, term.lemma_.lower(), term.label_) for term in final]
    df = pd.DataFrame(df, columns=["text", "lemma", "ent"])
    df = df.drop_duplicates()

    return df, final


def extract_terms_df(
    data,
    text_var,
    limit=1000,
    sample_size=3000,
    ngs=True,
    ents=True,
    ncs=True,
    drop_emoji=True,
    remove_punctuation=False,
    ngrams=(2, 2),
    include_pos=["NOUN", "PROPN", "ADJ"],
    include_types=["PERSON", "ORG"],
    language="en",
):
    data = data[data[text_var].notna()]
    data = data[text_var]
    data = data.drop_duplicates()
    data = data.sample(min(sample_size, len(data)))
    sentences = data.to_list()

    df_terms = pd.DataFrame()
    df_terms_spacy = []
    pbar = tqdm(total=len(sentences), desc="Extract Terms")
    for text in sentences:
        df, final = extract_terms(
            text,
            ngs=ngs,
            ents=ents,
            ncs=ncs,
            drop_emoji=drop_emoji,
            remove_punctuation=remove_punctuation,
            ngrams=ngrams,
            include_pos=include_pos,
            include_types=include_types,
            language=language,
        )
        df_terms = df_terms.append(df)
        df_terms_spacy.append(final)
        pbar.update(1)

    df_terms["text"] = df_terms["text"].apply(lambda x: x.strip())

    # entities
    entities = df_terms[["lemma", "ent"]].drop_duplicates()
    entities = entities[entities["ent"] != ""]
    entities = (
        entities.groupby("lemma")["ent"].apply(lambda x: " | ".join(x)).reset_index()
    )

    # Count terms
    count_terms = (
        df_terms.groupby(["lemma"]).agg(count_terms=("lemma", "count")).reset_index()
    )

    # Get the list of text
    df_drop = df_terms[["text", "lemma"]].drop_duplicates()
    text_list = (
        df_drop.groupby("lemma")["text"].apply(lambda x: " | ".join(x)).reset_index()
    )

    # Main Form
    main_form = (
        df_terms.groupby(["text"]).agg(count_text=("text", "count")).reset_index()
    )
    main_form = pd.merge(main_form, df_drop, on="text")
    main_form = main_form.sort_values(
        ["lemma", "text", "count_text"], ascending=(True, True, False)
    )
    main_form = main_form.drop("count_text", axis=1)
    main_form = main_form.groupby("lemma").head(1)
    main_form = main_form.rename(columns={"text": "main form"})
    main_form["main form"] = main_form["main form"].apply(
        lambda x: x.lower().capitalize()
    )

    # Create chi2 tablme
    terms_lemma = [[span.lemma_.lower() for span in doc] for doc in df_terms_spacy]
    doc_term_matrix, vocab = textacy.representations.build_doc_term_matrix(
        terms_lemma, tf_type="linear", idf_type="smooth"
    )
    df_chi2 = chi2_table(doc_term_matrix, vocab)

    full_list = pd.merge(count_terms, text_list, on="lemma")
    full_list = pd.merge(full_list, main_form, on="lemma")
    full_list = pd.merge(full_list, entities, on="lemma", how="left")
    full_list = pd.merge(full_list, df_chi2, on="lemma", how="left")

    full_list = full_list.sort_values("count_terms", ascending=False).reset_index(
        drop=True
    )
    full_list = full_list.head(limit)
    full_list = full_list[["lemma", "main form", "text", "ent", "count_terms", "chi2"]]

    return full_list
