from bunkatech.semantics.semantic_folding import semantic_folding, visualising_folding
import pandas as pd

folding_1 = ["problèmes", "danger", "menace"]
folding_2 = ["bienfaits", "opportunité", "génial", "innovation"]
semantic_group = [folding_1, folding_2]

data = pd.read_excel(
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/labeling/SHAI-LABELS-ROUND-1.xlsx"
)

embeddings = semantic_folding(
    semantic_group, data, text_var="title_lead", model="tfidf", dimension_folding=2
)

fig = visualising_folding(
    embeddings_folding=embeddings, data=data, text_var="title_lead", n_clusters=3
)
fig.show()
