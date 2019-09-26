from utils.additional_models import LabelEncoder
import torchvision.transforms as transforms
import requests
from settings import config
import numpy as np
from PIL import Image
from pickle import load
import io



def check_features(features: dict):
    FEATURES = sum(config["FEATURES"].values(), []) + config['ADDITIONAL_FEATURES']
    if features is not None and set(features.keys()) == set(FEATURES):
        cat_features = cast(
            [features.get(key) for key in config["FEATURES"]["CAT_FEATURES"]], int
        )
        num_features = cast(
            [features.get(key) for key in config["FEATURES"]["NUM_FEATURES"]], float
        )
        str_features = cast(
            [features.get(key).lower() for key in config["FEATURES"]["STR_FEATURES"]],
            str,
        )
        photos_url = features['photos']
        return (
            True,
            {
                "cat_features": cat_features,
                "num_features": num_features,
                "str_features": str_features,
                "photos": photos_url
            },
        )
    else:
        return False, 
        # return (
        #     False,
        #     f"You have not indicated these features - {set(features.keys())- set(FEATURES)}",
        # )


def cast(features, type):
    try:
        return np.array(list(map(type, features)), ndmin=2)
    except ValueError:
        pass    
        #raise AttributeError(f"{features} Must be {type} type")

def load_model(filename):
    with open(filename, "rb") as f:
        return load(f)


class ImageEmbedding:
    def __init__(self, model):
        self.model = model
        self.MODEL = load_model(config['MODELS']['embedding'])
        self.preprocess_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def get_feature_vector(self, photos):
        try:
            embedding = []
            for url in photos:
                image = self.download_image(url)
                input_image = Image.open(io.BytesIO(image))
                input_tensor = self.preprocess_image(input_image)
                input_batch = input_tensor.unsqueeze(0)  

                embedding.append(self.MODEL(input_batch).detach().numpy().mean(axis=2).mean(axis=2).flatten())
            return True, np.array(np.mean(embedding, axis=0), ndmin=2)
        except Exception as e:
            return False,

    def download_image(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            print('HTTPError: ', e)
        except requests.exceptions.ConnectionError as e:
            print('ConnectionError: ', e)
        except requests.exceptions.RequestException as e:
            print('RequestException: ', e)
        except requests.exceptions.Timeout as e:
            print('TimeoutError: ', e)

