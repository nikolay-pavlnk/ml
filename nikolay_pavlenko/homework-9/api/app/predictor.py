import numpy as np
from settings import config
import torch
from preprocessing import ImageEmbedding, load_model


class Predictor:
    def __init__(self):
        self.MODELS = {key: load_model(value) for key, value in config["MODELS"].items()}
        self.MODELS_PREPROC = {
            key: load_model(value) for key, value in config["UTIL_MODELS"].items()
        }
        self.MODELS_PREPROC_NEURAl = {key: load_model(value) for key, value in config["NEURAL_UTIL_MODELS"].items()}
        self.embeddor = ImageEmbedding(self.MODELS['embedding'])

    def predict(self, features, model):
        if model == "neural_network":
            is_valid, embedding = self.embeddor.get_feature_vector(features['photos'])
            if is_valid:      
                cluster = self.MODELS['kmean'].predict(embedding)
                features['cat_features'] = np.concatenate((features['cat_features'].flatten(), cluster))
                features['embedding'] = embedding
                x = torch.from_numpy(self.get_x(features, model)).float()
                response = str(self.MODELS[model](x).item())
            else:
                raise AttributeError
        else:
            response = str(self.MODELS[model].predict(self.get_x(features, model))[0])
        return response


    def get_x(self, features, model):
        x_str = self.MODELS_PREPROC["label_encoder_str"].transform(
            features.get("str_features")
        )
        if model == "neural_network":
            x_cat = self.MODELS_PREPROC_NEURAl["label_encoder_cat"].transform(
                features.get("cat_features")
            )
            x_one_hot = self.MODELS_PREPROC_NEURAl["one_hot"].transform(
                np.concatenate((x_str, x_cat), axis=1)
            )
            x = np.concatenate(
                (x_one_hot, np.array(features.get("num_features")), features['embedding']), axis=1
            )
            x = self.MODELS_PREPROC_NEURAl["minmax"].transform(x)
        else:
            x_cat = self.MODELS_PREPROC["label_encoder_cat"].transform(
                features.get("cat_features")
            )
            x = np.concatenate((x_str, x_cat, features.get("num_features")), axis=1)
        return x
