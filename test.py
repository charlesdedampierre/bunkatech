from bunkatech.semantics.indexer import indexer
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from random import sample
from bunkatech.search.fts5_search import fts5_search

docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
docs = sample(docs, 10000)

search_term = "weapon"

res = fts5_search(search_term, docs, case_sensitive=False)
