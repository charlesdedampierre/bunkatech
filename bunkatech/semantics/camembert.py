from multiprocessing import Pool
from fairseq.models.roberta import CamembertModel
import pandas as pd
from tqdm import tqdm

model_path = "camembert-base"
camembert = CamembertModel.from_pretrained(model_path)


"""

from transformers import CamembertModel, CamembertTokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
camembert = CamembertModel.from_pretrained("camembert-base")

# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("J'aime le camembert !")
encoded_sentence = tokenizer.encode(tokenized_sentence, max_length = 3000, truncation=False)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings = camembert(encoded_sentence)['last_hidden_state']
embeddings = embeddings.detach().numpy()[0]
final_embeddings = pd.DataFrame(embeddings).mean()
#embeddings, _ = camembert(encoded_sentence)"""


def camembert_sentence_embedding(sentence):

    # Extract the last layer's features
    tokens = camembert.encode(sentence)
    last_layer_features = camembert.extract_features(tokens)[0].detach().numpy()

    # The sentence is the mean of the embeddings
    return pd.DataFrame(last_layer_features).mean()


def camembert_embedding(texts):

    # parallelize the process
    with Pool(10) as p:
        res = list(tqdm(p.imap(camembert_sentence_embedding, texts), total=len(texts)))

    return pd.DataFrame(res)


if __name__ == "__main__":

    data = pd.read_excel(
        "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/labeling/SHAI-LABELS-ROUND-1.xlsx"
    )

    data["bindex"] = data.index
    data = data.sample(100).reset_index(drop=True)
    texts = data["title_lead"].values

    df_embeddings = camembert_embedding(texts)

    print(df_embeddings)
