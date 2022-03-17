def wrap_by_word(string, n_words):
    """returns a string where \\n is inserted between every n words"""
    try:
        a = string.split()
        ret = ""
        for i in range(0, len(a), n_words):
            ret += " ".join(a[i : i + n_words]) + "<br>"
    except:
        pass

    return ret

def f(x):
    return x*x


from fairseq.models.roberta import CamembertModel
from tqdm import tqdm
import pandas as pd


model_path = '/Volumes/OutFriend/camembert/camembert-base/'
camembert = CamembertModel.from_pretrained(model_path)

def camembert_embedding(list):
    embeddings = []
    for sentence in tqdm(list, total=len(list)):

        # Extract the last layer's features
        tokens = camembert.encode(sentence)
        last_layer_features = camembert.extract_features(tokens)[0].detach().numpy()

        # The sentence is the mean of the embeddings
        embedding = pd.DataFrame(last_layer_features).mean()
        embeddings.append(embedding)

    return embeddings
