from bunkatech.nested_topic_modeling import nested_topic_modeling
import pandas as pd
import plotly
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
df = pd.DataFrame(docs)
df.columns = ["text"]
df["bindex"] = df.index
df = df.sample(3000)

# Load Data
# df = pd.read_csv("data/imdb_movies.csv", index_col=[0])
treemap, sankey, sunburst = nested_topic_modeling(
    df,
    text_var="text",
    index_var="bindex",
    sample_size=2000,
    sample_terms=3000,
    embeddings_model="tfidf",
)
plotly.offline.plot(sankey, auto_open=True, filename="saved_graph/sankey.html")
plotly.offline.plot(treemap, auto_open=True, filename="saved_graph/treemap.html")
plotly.offline.plot(sunburst, auto_open=True, filename="saved_graph/sunburst.html")
