# BUNKAtech

<img src="images/bunka_logo.png" width="50%" height="50%" align="right" />

Bunkatech is an aggregation of machine learning modules specialized in the embedding of multi-format information.

The modules carry out semantic tasks using embeddings (terms extraction, document embeddings etc), Network analysis using Graph Embeddings, Search using Semantic Search, Trend Analysis using Moving Averages, Topic Modeling, Nested Topic Modeling, Image Embedding & Front-end display using Streamlit.

This README.md file has been created with the help of the fantastic work of [MaartenGr](https://github.com/MaartenGr) and his package [BERTopic](https://github.com/MaartenGr/BERTopic/blob/master/README.md)

## Installation

Create an environement using poetry to isolate the dependencies. This is very important given the fact that
bunkatech uses a lot of dependencies

```bash
poetry init
```

in pyproject.toml, set python = "^3.8" and the use the following command to use python 3.8 as an interpreter.
Then activate the environment.

```bash
poetry env use python3.8
poetry shell
```

Install the requirements using requirments.txt

```bash
pip install -r requirments.txt
```

Install some other specific packages

```bash
pip install sentence-transformers
pip install gensim
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_lg
```

Once everything is installed, clone the repository
Clone the repository and enter it. The cloning process may take few minutes.

```bash
git clone https://github.com/charlesdedampierre/bunkatech.git
cd bunkatech/
```

```python
import bunkatech
```

## Getting Started with real examples

For an in-depth overview of the features of BUNKAtech you can check the full documentation [here](https://docs.google.com/document/d/1CsJ-dhpm89e42hH7XPNuUtT1nAeCzC1kuIFEja_WyVs/edit) or you can follow along with one of the examples below using the Jupyter-Notebooks scripts contained in the examples/ repository.

## Quick Start with Bunka

```python
from bunkatech import Bunka
import pandas as pd

data = pd.read_csv('bunkatech/data/imdb.csv', index_col = [0])
data = data.sample(2000, random_state = 42)

# Instantiate the class. This will extract terms from the the text_var column, embed those terms and embed the documents.
bunka = Bunka(data = data,
                text_var = 'description',
                index_var = 'imdb',
                extract_terms=True,
                terms_embedding=True,
                docs_embedding=True,
                sample_size_terms=2000,
                terms_limit=2000,
                terms_ents=False,
                terms_ngrams=(2, 2),
                terms_ncs=False,
                terms_include_pos=["NOUN", "PROPN", "ADJ"],
                terms_include_types=["PERSON", "ORG"],
                terms_embedding_model="all-MiniLM-L6-v2",
                docs_embedding_model="all-MiniLM-L6-v2",
                language="en",
                terms_path=None,
                docs_dimension_reduction = 5,
                terms_embeddings_path=None,
                docs_embeddings_path=None,
                docs_multiprocessing = False,
                terms_multiprocessing = False)


# Extract the computed objects
terms = bunka.terms
terms_embeddings = bunka.terms_embeddings
docs_embeddings = bunka.docs_embeddings

```

Try out Some modules of Bunka

```python
fig = bunka.origami_projection_unique(
                    left_axis= ['love'],
                    right_axis = ['hate'],
                    height=800,
                    width=1500,
                    type="terms",
                    dispersion=True,
                    barometer=True,
                    explainer = True
    
                )
fig.show()
```

<img src="images/origami.png" width="60%" height="60%" align="center" />

The code above displays the projection of the terms on an axis 'love-hate' following the methodology contained in the following [paper](https://journals.sagepub.com/doi/full/10.1177/0003122419877135).

The methods traditionaly belongs to the class Origami but as Bunka inherited from this class, it can also call it.

```python
fig = bunka.fit_draw(
            variables=["main form"],
            top_n=500,
            global_filter=0.2,
            n_neighbours=6,
            method="node2vec",
            n_cluster=10,
            bin_number=30,
            black_hole_force=3,
            color="community",
            size="size",
            symbol="entity",
            textfont_size=9,
            edge_size=1,
            height=2000,
            width=2000,
            template="plotly_dark",
        )
```

<img src="images/networks.png" width="50%" height="50%" align="center" />

The code above calls a methods of the SemanticNetworks class. it creates a network of extracted terms using the [node2vec algorithm](https://snap.stanford.edu/node2vec/)

Display Nested Maps

```python
fig_nested = bunka.nested_maps(
                                size_rule="docs_size",
                                map_type="treemap", # Try sunburst
                                width=800,
                                height=800,
                                query=None) # You can query the map with an exact query

fig_nested.show()
```

<img src="images/nested.png" width="50%" height="50%" align="center" />

## Overview

The terms & embeddings are created when the function is initialized.
For quick access to common functions that use those embeddings, here is an overview of Bunkatech's main methods:

| Method | Code  |
|-----------------------|---|
| Project the Data on a Semantic Axis    |  `.origami_projection_unique(left_axis, right_axis)` |
| Create a Nested Map of the document Embeddings  |  `.nested_maps(map_type="sunburst")` |
| Get the list of 15 clusters described each by 5 terms    |  `.get_clusters(topic_number=15, top_terms = 5)` |
| Visualize the clusters with Plotly | `.visualize_topics_embeddings()`  |
| Get the centroids elements of each cluster     |  `.get_centroid_documents()` |
| Get the evolution of topics in time    |  `.temporal_topics()` |
| Get the Semantic Trend and the specific terms by trend |  `.moving_average_comparison()` |
| Draw a Semantic Network of terms based on co-occurence |  `.fit_draw()` |

## Calling Bunka on the Streamlit Package

Bunka modules can be used using the [Streamlit Package](https://streamlit.io/). All the code is located on the **app.py** script where you can decide of the data to ingest etc.

In order to call the platform locally on your machine.

```bash
streamlit run app.py
```

### Embeddings Models

Different embeddings modelds exist. They word with the Help of [Sentence-bert](https://www.sbert.net/). The better efficiency/time ratio is **all-MiniLM-L6-v2**. But when it comes to multilangual needs, distiluse-base-multilingual-cased-v1 works well.

### Parallel processing

By default the processus of terms extraction, terms embeddings & document embeddings are parralized to increase the speed.

More variables can me modified, they are all indicated in the description of the function.
