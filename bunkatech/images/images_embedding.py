import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import pickle
import os
import glob
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from random import sample


def extract_images(column_name: str, data: str, id: str = "id"):
    """Use Minet - Take a .csv(not tsv and create a dowloaded file with images)

    Args:
        data (str): data with the link to urls of images
        column_name (str): column with the link to the urls
    """
    cmd = f"minet fetch {column_name} {data} --filename {id} > report.csv"
    os.system(cmd)


def image_embedding(
    dir_images,
    destination_path,
    sample_size=500,
):
    """Make embeddings from the an image directory and save it as a pickle object

    Args:
        path_images ([type]): [description]
        destination_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    images = glob.glob(f"{dir_images}/*")
    images = sample(images, sample_size)
    print(len(images))

    # Load CLIP model
    model = SentenceTransformer("clip-ViT-B-32")

    pbar = tqdm(total=len(images))
    # Encode an image:
    embeddings = []
    images_worked = []
    for image in images:
        try:
            img_emb = model.encode(Image.open(image))
            embeddings.append(img_emb)
            images_worked.append(image)
            pbar.update(1)
        except:
            pbar.update(1)
            pass

    check_dir(destination_path)
    with open(destination_path + "/image_embedding.pickle", "wb") as f:
        pickle.dump(embeddings, f)

    with open(destination_path + "/images_worked.pickle", "wb") as f:
        pickle.dump(images_worked, f)

    return embeddings


def resize(images_path, destination_path):
    """Go to the directory names content and resize all images
    Put them in a directory called images_preprocessed"""

    check_dir(destination_path + "/images_preprocessed")

    # resize images
    path_images = glob.glob(images_path)
    pbar = tqdm(total=len(path_images))
    for image in path_images:
        try:
            im = Image.open(image)
            imResize = im.resize((200, 200), Image.ANTIALIAS)
            image_name = image.split("/downloaded/")[1]
            imResize.save(destination_path + "/images_preprocessed/" + image_name)
            pbar.update(1)
        except:
            print("error")
            pbar.update(1)
            pass


def check_dir(path):
    """
    This functions checks if a directory exists; if not, it creates it.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getImage(path):
    return OffsetImage(plt.imread(path))


def visualize(
    embeddings: list,
    path_images: list,
    destination_path: str,
    dpi: int = 100,
    figsize: tuple = (120, 120),
):
    """from embedding to images"""

    print("Reduction Algorithm...")
    tsne_data = TSNE(n_components=2).fit_transform(embeddings)
    df = pd.DataFrame(tsne_data)
    df.columns = ["x", "y"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df.x, df.y)

    for x0, y0, path in zip(df.x, df.y, path_images):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)

    plt.savefig(destination_path + "/image.png", dpi=dpi)


if __name__ == "__main__":
    """resize(
        images_path="/Volumes/OutFriend/imaginary world/images/downloaded/*",
        destination_path="/Volumes/OutFriend/imaginary world/images",
    )
    """

    """image_embedding(
        sample_size=1000,
        dir_images="/Volumes/OutFriend/imaginary world/images/images_preprocessed",
        destination_path="/Volumes/OutFriend/imaginary world/images/embeddings",
    )"""

    with open(
        "/Volumes/OutFriend/imaginary world/images/embeddings/image_embedding.pickle",
        "rb",
    ) as handle:
        embeddings = pickle.load(handle)

    with open(
        "/Volumes/OutFriend/imaginary world/images/embeddings/images_worked.pickle",
        "rb",
    ) as handle:
        path_images = pickle.load(handle)

    visualize(
        embeddings,
        path_images,
        destination_path="/Volumes/OutFriend/imaginary world/images/embeddings",
        figsize=(220, 220),
    )
