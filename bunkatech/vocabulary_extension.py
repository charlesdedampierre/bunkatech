import pandas as pd
from gensim.models import Word2Vec as wv


def erase_comma(x):
    try:
        res = x.split(",")[0]
    except:
        res = x
    return res


def erase_point(x):
    try:
        res = x.split(".")[0]
    except:
        res = x
    return res


def erase_accent(x):
    try:
        res = x.split("'")[1]
    except:
        res = x
    return res


data = pd.read_excel(
    "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/labeling/SHAI-LABELS-ROUND-1.xlsx"
)

text_var = "title_lead"
folding = ["voiture"]

docs = data[text_var].to_list()
final = [doc.split() for doc in docs]
model = wv(
    sentences=final, vector_size=100, window=10, min_count=3, workers=8, epochs=15
)

df_final = pd.DataFrame()

for word in folding:
    try:
        res = model.wv.most_similar(word, topn=10)
        df = pd.DataFrame(res, columns=["similar_words", "score"])
        df["word"] = word
        df_final = df_final.append(df)
    except:
        pass

df_final = df_final[["word", "similar_words", "score"]]
df_final["similar_words"] = df_final["similar_words"].apply(lambda x: x.lower())
df_final["similar_words"] = df_final["similar_words"].apply(lambda x: erase_comma(x))
df_final["similar_words"] = df_final["similar_words"].apply(lambda x: erase_point(x))
df_final["similar_words"] = df_final["similar_words"].apply(lambda x: erase_accent(x))

print(df_final)
