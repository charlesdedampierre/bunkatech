# BUNKAtech

## Bunkatech Overall

Bunkatech is an aggregation of machine learning modules carrying out a set of diverse tasks.

**bunka_class.py** contains the **Bunka** class that inherits from sublass carrying out specific tasks.

The list of those subclasses is the following:

- **NestedTopicModeling** : nested topic modeling and creation of nested maps (treemaps, sunburst, sankey diagrams, icicle)
- **Origami** : Create Origami projection, ie after defining axes with words, it projects terms or documents on those axis
- **SemanticsTrends** : create trends taking into account semantics
- **SemanticNetworks** : create semantic networks
- **TopicModeling** : simple topic modeling

All those subclasses inherit form a base class called **BasicsSemantics**

## Start with bunkatech

first, install the necessary packages:

```bash
pip install requirements.txt
```

In order install the package locally on your marchine, creates the python package using setup.py and install it trought pip in your local package repository.

```bash
python setup.py sdist bdist_wheel
cd dist/
pip install bunkatech-0.0.1.tar.gz
```

## Quick Start with Basic Semantics

The Basic Semantic class carries out the basic semantic actions of Bunka:

- terms extraction
- terms embeddings
- documents embeddings

A terms refers to a short or unique combination of words. A documents (or docs) refers to a sentence, a paragraph or even a full article or abstract.

```python

from bunkatech import BasicSemantics
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
data = pd.DataFrame(docs, columns = ['docs'])
data = data.sample(2000, random_state = 42)
data['doc_index'] = data.index


model = BasicSemantics(data, text_var = 'docs', index_var = 'doc_index')

model.fit(extract_terms=True,
        terms_embedding=True,
        docs_embedding=True,
        sample_size_terms=2000,
        terms_limit=5000,
        terms_ents=False,
        terms_ngrams=(2, 2),
        terms_ncs=False,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        terms_embedding_model="all-MiniLM-L6-v2",
        docs_embedding_model="all-MiniLM-L6-v2",
        language="en",
        terms_path=None,
        terms_embeddings_path=None,
        docs_embeddings_path=None,
        docs_dimension_reduction=5
        language = 'en')

# Extract the computed objects
terms = model.terms
terms_embeddings = model.terms_embeddings
docs_embeddings = model.docs_embeddings

```

The class instantiates itself with a pandas DataFrame and the name of the text_var columns & index_var columns.

### Embeddings Models

Different embeddings modelds exist. They word with the Help of [Sentence-bert](https://www.sbert.net/). The better efficiency/time ratio is **all-MiniLM-L6-v2**. But when it comes to multilangual needs, distiluse-base-multilingual-cased-v1 works well.

### Parallel processing

By default the processus of terms extraction, terms embeddings & document embeddings are parralized to increase the speed.

More variables can me modified, they are all indicated in the description of the function.

## Quick Start with Bunka

```python
from bunkatech import Bunka
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
data = pd.DataFrame(docs, columns = ['docs'])
data = data.sample(2000, random_state = 42)
data['doc_index'] = data.index


bunka = Bunka(data, 
            text_var = 'docs', 
            index_var = 'doc_index', 
            extract_terms=True,
            terms_embedding=False,
            docs_embedding=False)


# Extract the computed objects
terms = bunka.terms
terms_embeddings = bunka.terms_embeddings
docs_embeddings = bunka.docs_embeddings

```

As the class inherits from **BasicSemantics**, it also instantiates itself with a pandas DataFrame and the name of the text_var columns & index_var columns. There is a lighter fitted part.

As the class also inherits from other sublasses, we can call also their methods without recodign everything in the Bunka class.

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

The code above calls a methods of the SemanticNetworks class. it creates a network of extracted terms using the [node2vec algorithm](https://snap.stanford.edu/node2vec/)

## Calling Bunka on the Streamlit Package

Bunka modules can be used using the [Streamlit Package](https://streamlit.io/). All the code is located on the **app.py** script where you can decide of the data to ingest etc.

In order to call the platform locally on your machine.

```bash
streamlit run app.py
```
