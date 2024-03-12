import pickle
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)

INPUT_DATA_PATH = Path("../data/winequality-white.csv")
TRAIN_DATASET_PATH = Path("./train_dataset.pkl")
TEST_DATASET_PATH = Path("./test_dataset.pkl")
OUTPUT_MODEL_PATH = Path("./model.pth")

N_EPOCHS = 8
BATCH_SIZE = 32

# The wine attributes
N_INPUT_FEATURES = 11

# 11 output scores representing integers from [0, 9]
N_OUTPUT_FEATURES = 10


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the network
        self.nn_stack = nn.Sequential(
            nn.Linear(N_INPUT_FEATURES, 100),
            nn.ReLU(),
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, N_OUTPUT_FEATURES),
            # Softmax along the 1st dimension (column)
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        probabilities = self.nn_stack(x)
        return probabilities


def train_epoch(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # Set the `model` to be in training mode
    model.train()

    # Iterate the batches from the `dataloader`
    for batch_id, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute the prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Back-propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_id % 20 == 0:
            lossScore = loss.item()
            currentProcessed = (batch_id + 1) * len(X)
            print(f"loss: {lossScore:>7f}  [{currentProcessed:>5d}/{size:>5d}]")


def test_epoch(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    test_loss = 0
    correct_count = 0

    # Disable gradient computation since we are not back-propagating on this data
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct_count += (
                (torch.argmax(pred, dim=1) == torch.argmax(y, dim=1))
                .type(torch.int)
                .sum()
                .item()
            )

    # Normalize the loss on a per-batch basis
    test_loss /= n_batches
    correct_count /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct_count):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def preprocess_data():
    # Load the data
    dataset_df = pd.read_csv(INPUT_DATA_PATH, sep=";")
    X_df = dataset_df[
        [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ]
    ]
    y_df = dataset_df["quality"]

    # Convert the X, y dataframes to respective tensors. Convert the `y_df` to
    # a one-hot-encoded counterpart
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)

    # The classes are 0 indexed
    num_classes = y_df.max() + 1
    y_tensor = torch.eye(num_classes)[y_df]

    # Convert the X, y tensors into a single feature, target `dataset`
    dataset = TensorDataset(X_tensor, y_tensor)

    # Perform an 80-20 train-test split on the `dataset`
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    # Write the train/test datasets to file via pickle
    with open(TRAIN_DATASET_PATH, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(TEST_DATASET_PATH, "wb") as f:
        pickle.dump(test_dataset, f)


def main():
    # Get cpu, gpu or mps device for training.
    device = get_device()
    print(f"Using {device} device")

    # Check that the respective datasets exist
    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError("Train dataset does not exist.")

    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError("Test dataset does not exist.")

    # Load the train/test datasets
    with open(TRAIN_DATASET_PATH, "rb") as f:
        train_dataset = pickle.load(f)

    with open(TEST_DATASET_PATH, "rb") as f:
        test_dataset = pickle.load(f)

    # Create the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Instantiate the `model`. It is important to instantiate the model on
    # the target device before the `optimizer` below
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-03)

    # Train the model!
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch}\n")
        print("-" * 30 + "\n")
        train_epoch(device, train_dataloader, model, loss_fn, optimizer)
        test_epoch(device, test_dataloader, model, loss_fn)

    print("Done!")

    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"Saved PyTorch Model State to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    preprocess_data()
    main()
