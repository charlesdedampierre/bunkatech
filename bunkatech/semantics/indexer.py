import sqlite3
import pandas as pd
import os


def indexer(docs: list, terms: list, db_path = '/Volumes/OutFriend'):
    
    terms = [x for x in terms if not x.startswith('"')]
    terms = [x for x in terms if not x.endswith('"')]
    terms = [x for x in terms if not x.startswith("'")]
    terms = [x for x in terms if not x.endswith("'")]
    terms = [x.replace('"', '') for x in terms]
    terms = ['"'+x+'"' for x in terms]

    conn = sqlite3.connect(db_path + "/database.db")

    # insert the docs
    df_docs = pd.DataFrame(docs, columns=["data"])
    df_docs["neo_id"] = df_docs.index
    df_docs.to_sql("docs", conn, if_exists="replace", index=False)

    # insert the terms
    df_terms = pd.DataFrame(terms, columns=["words"])
    df_terms.to_sql("terms", conn, if_exists="replace", index=False)

    # index
    idfield = "neo_id"
    field = "data"
    table = "docs"
    terms = "terms"
    words = "words"

    # conn.enable_load_extension(True)
    c = conn.cursor()

    # define the name of the ouput column
    output = "indexed_{}".format(terms)

    c.execute("DROP table IF EXISTS abstractsearch;")
    c.execute("DROP table IF EXISTS uniqueterms;")
    c.execute("DROP table IF EXISTS " + output + ";")
    conn.commit()

    c.execute("CREATE VIRTUAL TABLE abstractsearch USING fts5(neo_id, data);")
    conn.commit()
    c.execute(
        "INSERT INTO abstractsearch SELECT "
        + idfield
        + " AS neo_id, "
        + field
        + " AS data FROM "
        + table
        + ";"
    )

    conn.commit()

    c.execute(
        "CREATE TABLE uniqueterms AS SELECT DISTINCT "
        + words
        + " AS words FROM "
        + terms
        + ";"
    )
    conn.commit()

    c.execute(
        "CREATE TABLE "
        + output
        + " AS SELECT * FROM abstractsearch, uniqueterms WHERE abstractsearch.data MATCH uniqueterms.words COLLATE NOCASE;"
        ""
    )
    conn.commit()

    c.execute("DROP table IF EXISTS abstractsearch;")
    c.execute("DROP table IF EXISTS uniqueterms;")

    df_docs_table = pd.read_sql_query("SELECT * FROM docs", conn)
    df_docs_table = df_docs_table.rename(columns={"data": "docs"})

    df_indexed_table = pd.read_sql_query("SELECT * FROM indexed_terms", conn)
    final = pd.merge(df_docs_table, df_indexed_table, on="neo_id")
    final = final[["docs", "words"]].copy()
    
    final['words'] = final['words'].apply(lambda x: x.strip('"'))

    os.remove(db_path + "/database.db")

    return final
