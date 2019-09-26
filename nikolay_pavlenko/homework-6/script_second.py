import torch
from torch.utils.data import DataLoader
import argparse
import pathlib
import copy
import json

from utilities import CsvReader, classification_report, regression_report, LoadDataset
from models_torch import LinearRegressionTorch, LogisticRegressionTorch, BaseLinear


def estimate(problem, path_train, path_test, target, cuda, config):
    csvreader_train = CsvReader(target, path_train, 5)
    csvreader_test = CsvReader(target, path_test, 5)
    loader_train = LoadDataset()
    loader_train.fit(csvreader_train)
    loader_test = copy.deepcopy(loader_train)

    loader_train.transform(csvreader_train)
    train_data = DataLoader(dataset=loader_train, batch_size=config["batch_size"])
    loader_test.transform(csvreader_test)
    test_data = DataLoader(dataset=loader_test, batch_size=config["batch_size"])

    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if problem == "classification":
        estimator = LogisticRegressionTorch(input_dim=loader_train.dim, device=device)
        report = classification_report
        criterion = torch.nn.BCELoss()
    elif problem == "regression":
        estimator = LinearRegressionTorch(input_dim=loader_train.dim, device=device)
        report = regression_report
        criterion = torch.nn.MSELoss()

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(estimator.parameters())
    else:
        optimizer = torch.optim.SGD(estimator.parameters(), lr=config["learning_rate"])

    if config.get("epochs", False):
        epochs = config["epochs"]
    else:
        epochs = 1e4

    trainer = BaseLinear(estimator, device, criterion, optimizer, epochs)
    trainer.fit(train_data)

    X_test, y_test = next(iter(test_data))
    X_test, y_test = X_test.to(device), y_test.to(device)

    X_train, y_train = next(iter(train_data))
    X_train, y_train = X_train.to(device), y_train.to(device)

    print("Train metrics\n-------")
    report(y_train[:, None], estimator(X_train))
    print("Test metrics\n-------")
    report(y_test[:, None], estimator(X_test))


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
