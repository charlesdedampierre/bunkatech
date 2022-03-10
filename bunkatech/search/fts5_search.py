import sqlite3
import pandas as pd
import os


def fts5_search(
    search_terms: list, docs: list, case_sensitive: bool = False
) -> pd.DataFrame:
    """The function search a query in a list of documents and outputs
    all the documents containing the the query

    Parameters
    ----------
    search_term : str
    docs : list
        Documents to search from

    case_sensitive: bool
        indicate if the case matters

    Returns
    -------
    pd.DataFrame
        Documents containing the query
    """

    terms = [x for x in search_terms if not x.startswith('"')]
    terms = [x for x in terms if not x.endswith('"')]
    terms = [x for x in terms if not x.startswith("'")]
    terms = [x for x in terms if not x.endswith("'")]
    terms = [x.replace('"', "") for x in terms]
    terms = ['"' + x + '"' for x in terms]

    db_path = "."
    conn = sqlite3.connect(db_path + "/search_database.db")
    c = conn.cursor()

    # Insert the documents to search from in a sqlite3 database
    df_docs = pd.DataFrame(docs, columns=["data"])
    df_docs["neo_id"] = df_docs.index
    df_docs.to_sql("docs", conn, if_exists="replace", index=False)

    # insert the search term(s)
    df_terms = pd.DataFrame(terms, columns=["words"])
    df_terms.to_sql("terms", conn, if_exists="replace", index=False)

    # conn.enable_load_extension(True)
    c = conn.cursor()

    # Starts the FTS5 abstract Search
    c.execute("CREATE VIRTUAL TABLE abstractsearch USING fts5(neo_id, data);")
    conn.commit()

    # Insert the documents to serach terms from
    c.execute(
        "INSERT INTO abstractsearch SELECT neo_id AS neo_id, data AS data FROM docs;"
    )
    conn.commit()

    # Create a unique terms table based on the terms table
    c.execute("CREATE TABLE uniqueterms AS SELECT DISTINCT words AS words FROM terms;")
    conn.commit()

    # Create the output tabke with the result of the TFS5 Search

    if case_sensitive is True:
        c.execute(
            "CREATE TABLE indexed_terms AS SELECT neo_id, words FROM abstractsearch, uniqueterms WHERE abstractsearch.data MATCH uniqueterms.words;"
            ""
        )

    else:
        c.execute(
            "CREATE TABLE indexed_terms AS SELECT neo_id, words FROM abstractsearch, uniqueterms WHERE abstractsearch.data MATCH uniqueterms.words COLLATE NOCASE;"
            ""
        )

    conn.commit()

    c.execute("DROP table IF EXISTS abstractsearch;")
    c.execute("DROP table IF EXISTS uniqueterms;")

    df_docs_table = pd.read_sql_query("SELECT * FROM docs", conn)
    df_docs_table = df_docs_table.rename(columns={"data": "docs"})

    df_indexed_table = pd.read_sql_query("SELECT * FROM indexed_terms", conn)
    final = pd.merge(df_docs_table, df_indexed_table, on="neo_id")
    final["words"] = final["words"].apply(lambda x: x.strip('"'))
    final = final.drop("neo_id", axis=1)

    os.remove(db_path + "/search_database.db")

    return final
