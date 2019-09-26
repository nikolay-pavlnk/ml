import numpy as np
import torch
import argparse
import pathlib
import json

from model_selection import OneHotEncoder, StandardScaler
from utilities import CsvReader, classification_report, regression_report
from models_own import LinearRegression, LogisticRegression


def estimate(problem, path_train, path_test, target, cuda, config):
    csvreader_train = CsvReader(target, path_train, 5)
    csvreader_test = CsvReader(target, path_test, 5)
    X_train, y_train = csvreader_train.get_X_y()
    X_test, y_test = csvreader_test.get_X_y()

    one_hot = OneHotEncoder()
    scaler = StandardScaler()

    one_hot.fit(X_train[:, csvreader_train.categorical_idx])
    X_train_one_hot = one_hot.transform(X_train[:, csvreader_train.categorical_idx])
    X_test_one_hot = one_hot.transform(X_test[:, csvreader_test.categorical_idx])

    X_train_new = np.hstack(
        (X_train_one_hot, X_train[:, csvreader_train.continuous_idx])
    )
    X_test_new = np.hstack((X_test_one_hot, X_test[:, csvreader_test.continuous_idx]))

    scaler.fit(X_train_new)
    X_train_scaled = scaler.transform(X_train_new)
    X_test_scaled = scaler.transform(X_test_new)

    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    X_train_scaled = torch.from_numpy(X_train_scaled).to(device)
    X_test_scaled = torch.from_numpy(X_test_scaled).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    if problem == "classification":
        estimator = LogisticRegression(**config)
        report = classification_report
    elif problem == "regression":
        estimator = LinearRegression(**config)
        report = regression_report

    estimator.fit(X_train_scaled, y_train)
    print("Train metrics\n-------")
    report(y_train, estimator.predict(X_train_scaled))
    print("Test metrics\n-------")
    report(y_test, estimator.predict(X_test_scaled))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        help="type of task (classification or regression)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train", help="path to the train dataset", type=pathlib.Path, required=True
    )
    parser.add_argument(
        "--test", help="path to the test dataset", type=pathlib.Path, required=True
    )
    parser.add_argument("--target", help="name of target", type=str, required=True)
    parser.add_argument(
        "--config",
        help="path to the configuration file",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--cuda", help="if you need to use gpu", type=bool, default=False
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    estimate(args.problem, args.train, args.test, args.target, args.cuda, config)
