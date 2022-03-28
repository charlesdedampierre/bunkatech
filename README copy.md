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

The class instantiates itself with a pandas DataFrame and the name of the `text_var` columns & `index_var` columns.
