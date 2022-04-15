import pandas as pd
import umap
from sklearn.manifold import TSNE

path = (
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/content/"
)

data = pd.read_csv(path + "/semb.csv", index_col=[0])

print(data)

"""red = TSNE(n_components=2, learning_rate="auto", init="random")

# red = umap.UMAP(n_components=3, verbose=True)
# red.fit(data)
res = red.fit_transform(data)
df_res = pd.DataFrame(res)
df_res.index = data.index

df_res.to_csv(path + "/semb_tsen_2d.csv")
"""
