from fairseq.models.roberta import CamembertModel
from tqdm import tqdm
import pandas as pd

camembert = CamembertModel.from_pretrained(
    "/Volumes/OutFriend/camembert/camembert-base/"
)


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
