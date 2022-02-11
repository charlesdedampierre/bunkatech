from bunkatech.nested_topic_modeling import nested_topic_modeling
import pandas as pd
import plotly

# Load Data
df = pd.read_csv("data/imdb_movies.csv", index_col=[0])
treemap, sankey, sunburst = nested_topic_modeling(
    df,
    text_var="description",
    index_var="imdb",
    sample_size=2000,
    sample_terms=3000,
    embeddings_model="sbert",
)
plotly.offline.plot(sankey, auto_open=True, filename="saved_graph/sankey.html")
plotly.offline.plot(treemap, auto_open=True, filename="saved_graph/treemap.html")
plotly.offline.plot(sunburst, auto_open=True, filename="saved_graph/sunburst.html")
