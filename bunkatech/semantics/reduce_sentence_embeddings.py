import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

path = (
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/content"
)
data = pd.read_csv(path + "/semb_direct_pca_50d.csv", index_col=[0])

# red = TSNE(n_components=2, verbose=1, metric="cosine", perplexity=15, n_iter=3000)

red = umap.UMAP(n_components=2, verbose=True)

red.fit(data)
res = red.fit_transform(data)
df_res = pd.DataFrame(res)
df_res.index = data.index


"""
red = PCA(n_components=50)
red.fit(data)
res = red.transform(data)
df_res = pd.DataFrame(res)
df_res.index = data.index
df_res.to_csv(path + "/semb_direct_pca_50d.csv")

"""

df_res.to_csv(path + "/semb_direct_umap_2d.csv")

# df_res.to_csv(path + "/semb_direct_tsne_2d.csv")
