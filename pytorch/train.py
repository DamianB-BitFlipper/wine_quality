import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

INPUT_DATA_PATH = Path("../data/winequality-white.csv")
TRAIN_DATASET_PATH = Path("./train_dataset.pkl")
VALIDATE_DATASET_PATH = Path("./validate_dataset.pkl")
TEST_DATASET_PATH = Path("./test_dataset.pkl")
OUTPUT_MODEL_PATH = Path("./model.pth")

N_EPOCHS = 48
BATCH_SIZE = 32

# The wine attributes
N_INPUT_FEATURES = 11

# The output regression
N_OUTPUT_FEATURES = 1


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def train_validate_test_split(df, train_size, validate_size, random_state=42):
    train, validate, test = np.split(
        df.sample(frac=1, random_state=random_state),
        [int(train_size * len(df)), int((train_size + validate_size) * len(df))],
    )
    return train, validate, test


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the network components
        self.linear1 = nn.Linear(N_INPUT_FEATURES, 20)
        self.linear2 = nn.Linear(20, N_OUTPUT_FEATURES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        # Flatten the tensor to 1-D vector of `BATCH_SIZE`
        x = x.view(-1)

        return x


def train_epoch(device, dataloader, model, loss_fn, optimizer):
    # Set the `model` to be in training mode
    model.train()

    avg_loss = 0.0

    # Iterate the batches from the `dataloader`
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute the prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Back-propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item()

    print(f"loss: {avg_loss / len(dataloader):>7f}")


def test_epoch(device, dataloader, model, loss_fn, *, error_type_str):
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
            correct_count += (pred.round() == y).type(torch.int).sum().item()

    # Normalize the loss on a per-batch basis
    test_loss /= n_batches
    correct_count /= size
    print(
        f"{error_type_str} Error: \n Accuracy: {(100*correct_count):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def preprocess_data():
    # Load the data
    dataset_df = pd.read_csv(INPUT_DATA_PATH, sep=";")

    train_df, validate_df, test_df = train_validate_test_split(
        dataset_df, train_size=0.9, validate_size=0.03
    )

    X_cols = [
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

    y_col = "quality"

    # Pull out the relevant values into the respective X, y dataframes
    X_train_df, y_train_series = train_df[X_cols], train_df[y_col]
    X_validate_df, y_validate_series = validate_df[X_cols], validate_df[y_col]
    X_test_df, y_test_series = test_df[X_cols], test_df[y_col]

    # Standardize the inputs to mean 0, std 1. Importantly use the training dataset only!
    mean = X_train_df.mean()
    std = X_train_df.std()

    X_train_df = (X_train_df - mean) / std
    X_validate_df = (X_validate_df - mean) / std
    X_test_df = (X_test_df - mean) / std

    # Convert the X, y dataframes to respective tensors
    X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_series.values, dtype=torch.float32)  # type: ignore
    X_validate_tensor = torch.tensor(X_validate_df.values, dtype=torch.float32)
    y_validate_tensor = torch.tensor(y_validate_series.values, dtype=torch.float32)  # type: ignore
    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_series.values, dtype=torch.float32)  # type: ignore

    # Convert the X, y tensors into feature-target `dataset`s
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    validate_dataset = TensorDataset(X_validate_tensor, y_validate_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Sanity checks
    assert len(train_dataset) != 0
    assert len(validate_dataset) != 0
    assert len(test_dataset) != 0

    # Write the train/validate/test datasets to file via pickle
    with open(TRAIN_DATASET_PATH, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(VALIDATE_DATASET_PATH, "wb") as f:
        pickle.dump(validate_dataset, f)

    with open(TEST_DATASET_PATH, "wb") as f:
        pickle.dump(test_dataset, f)


def main():
    # Get cpu, gpu or mps device for training.
    device = get_device()
    print(f"Using {device} device")

    # Check that the respective datasets exist
    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError("Train dataset does not exist.")

    if not VALIDATE_DATASET_PATH.exists():
        raise FileNotFoundError("Validate dataset does not exist.")

    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError("Test dataset does not exist.")

    # Load the train/validate/test datasets
    with open(TRAIN_DATASET_PATH, "rb") as f:
        train_dataset = pickle.load(f)

    with open(VALIDATE_DATASET_PATH, "rb") as f:
        validate_dataset = pickle.load(f)

    with open(TEST_DATASET_PATH, "rb") as f:
        test_dataset = pickle.load(f)

    # Create the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    validate_dataloader = DataLoader(validate_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Instantiate the `model`. It is important to instantiate the model on
    # the target device before the `optimizer` below
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)

    # Train the model!
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch}\n")
        print("-" * 30 + "\n")
        train_epoch(device, train_dataloader, model, loss_fn, optimizer)
        test_epoch(
            device, validate_dataloader, model, loss_fn, error_type_str="Validate"
        )

    # Evaluate the final model on the `test_dataset`
    test_epoch(device, test_dataloader, model, loss_fn, error_type_str="Test")

    print("Done!")

    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"Saved PyTorch Model State to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    preprocess_data()
    main()

# TODO to make this better
# 1. Use a binomial distribution to transform the output between 0,9
# 3. Play with a different model architecture
# 5. Early stopping to prevent over-fitting
#    - Make train,validate,test datasets
#    - Eval on the validate during fitting
#    - If no significant improvements in validation error for 5 epochs, stop training
#    - Eval on the test at the end
# 6. Play with learning rates
# 7. Dropout, batchnorms
# 8. Learning rate schedule, large -> small
# 9. Try adding a regularization term to the Loss Fn
