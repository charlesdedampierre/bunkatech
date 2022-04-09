import pandas as pd
import glob
from tqdm import tqdm
import umap
import pickle
from sklearn.decomposition import PCA

"""path = "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/content/terms_embeddings"

files = glob.glob(path + "/*")

data = pd.DataFrame()
for file in tqdm(files, total=len(files)):
    dat = pd.read_csv(file, index_col=[0])
    data = data.append(dat)

data.to_csv(
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/content/terms_embeddings.csv"
)"""


path = (
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/content/"
)
data = pd.read_csv(
    path + "/terms_embeddings.csv",
    index_col=[0],
)

# red = umap.UMAP(n_components=10, verbose=True)
# Use PCA to reduce dimensions to 30-50;
# Then use UMAP to reduce to 10;
red = PCA(n_components=50)
red.fit(data)
res = red.transform(data)
df_res = pd.DataFrame(res)
df_res.index = data.index

df_res.to_csv(path + "/terms_embeddings_pca.csv")

with open(
    path + "/pca_model.pickle",
    "wb",
) as handle:
    pickle.dump(red, handle, protocol=pickle.HIGHEST_PROTOCOL)
