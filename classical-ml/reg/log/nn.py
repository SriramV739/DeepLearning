import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Make results reproducible
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. Read the dataset
# ============================================================

df = pd.read_csv("logistic_regression_data.csv")

label = "label"
train_cols = [column for column in df.columns if column != label]

X = df[train_cols].values.astype(np.float32)
y = df[label].values.astype(np.float32).reshape(-1, 1)

input_dim = X.shape[1]

print("Number of rows:", len(X))
print("Number of input features:", input_dim)


# ============================================================
# 3. Split into training, validation, and test data
#
# 64% training
# 16% validation
# 20% testing
# ============================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42,
    stratify=y_train_val
)


# ============================================================
# 4. Normalize the input features
#
# Only fit the scaler on the training data.
# Labels remain unchanged as 0 and 1.
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ============================================================
# 5. Convert everything to PyTorch tensors
# ============================================================

train_x_tensor = torch.tensor(X_train, dtype=torch.float32)
train_y_tensor = torch.tensor(y_train, dtype=torch.float32)

val_x_tensor = torch.tensor(X_val, dtype=torch.float32)
val_y_tensor = torch.tensor(y_val, dtype=torch.float32)

test_x_tensor = torch.tensor(X_test, dtype=torch.float32)
test_y_tensor = torch.tensor(y_test, dtype=torch.float32)


# ============================================================
# 6. Flexible neural network class
# ============================================================

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes,
        activation_name="relu",
        dropout=0.0
    ):
        super().__init__()

        layers = []
        previous_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(
                nn.Linear(previous_size, hidden_size)
            )

            if activation_name == "relu":
                layers.append(nn.ReLU())

            elif activation_name == "tanh":
                layers.append(nn.Tanh())

            elif activation_name == "gelu":
                layers.append(nn.GELU())

            else:
                raise ValueError(
                    f"Unknown activation: {activation_name}"
                )

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            previous_size = hidden_size

        # Final layer produces one number called a logit.
        # Sigmoid is handled by BCEWithLogitsLoss.
        layers.append(nn.Linear(previous_size, 1))

        self.neural_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.neural_net(x)


# ============================================================
# 7. Architectures and configurations to test
# ============================================================

configurations = [
    {
        "name": "Logistic regression",
        "hidden_sizes": [],
        "activation": "relu",
        "dropout": 0.0,
        "learning_rate": 0.001
    },
    {
        "name": "One small hidden layer",
        "hidden_sizes": [6],
        "activation": "relu",
        "dropout": 0.0,
        "learning_rate": 0.001
    },
    {
        "name": "Your original architecture",
        "hidden_sizes": [6, 4],
        "activation": "relu",
        "dropout": 0.0,
        "learning_rate": 0.001
    },
    {
        "name": "Two medium hidden layers",
        "hidden_sizes": [16, 8],
        "activation": "relu",
        "dropout": 0.0,
        "learning_rate": 0.001
    },
    {
        "name": "Two equal hidden layers",
        "hidden_sizes": [16, 16],
        "activation": "relu",
        "dropout": 0.0,
        "learning_rate": 0.001
    },
    {
        "name": "Wide network",
        "hidden_sizes": [32, 16],
        "activation": "relu",
        "dropout": 0.0,
        "learning_rate": 0.001
    },
    {
        "name": "Deep network",
        "hidden_sizes": [32, 16, 8],
        "activation": "relu",
        "dropout": 0.0,
        "learning_rate": 0.001
    },
    {
        "name": "Medium network with dropout",
        "hidden_sizes": [16, 8],
        "activation": "relu",
        "dropout": 0.2,
        "learning_rate": 0.001
    },
    {
        "name": "Medium network with Tanh",
        "hidden_sizes": [16, 8],
        "activation": "tanh",
        "dropout": 0.0,
        "learning_rate": 0.001
    },
    {
        "name": "Medium network with GELU",
        "hidden_sizes": [16, 8],
        "activation": "gelu",
        "dropout": 0.0,
        "learning_rate": 0.001
    }
]


# ============================================================
# 8. Loss function
# ============================================================

loss_fn = nn.BCEWithLogitsLoss()


# ============================================================
# 9. Function to evaluate a model
# ============================================================

def evaluate_model(model, x_tensor, y_tensor):
    model.eval()

    with torch.no_grad():
        logits = model(x_tensor)

        loss = loss_fn(
            logits,
            y_tensor
        )

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

        accuracy = (
            predictions == y_tensor
        ).float().mean()

    return loss.item(), accuracy.item()


# ============================================================
# 10. Function to train one model
#
# Early stopping keeps the model state that had the lowest
# validation loss.
# ============================================================

def train_model(
    configuration,
    max_epochs=10000,
    patience=300
):
    torch.manual_seed(42)

    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_sizes=configuration["hidden_sizes"],
        activation_name=configuration["activation"],
        dropout=configuration["dropout"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configuration["learning_rate"]
    )

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    best_model_state = None

    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Training mode enables dropout, if present.
        model.train()

        logits = model(train_x_tensor)

        training_loss = loss_fn(
            logits,
            train_y_tensor
        )

        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        # Check validation performance after this update.
        validation_loss, validation_accuracy = evaluate_model(
            model,
            val_x_tensor,
            val_y_tensor
        )

        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Restore the weights that gave the best validation loss.
    model.load_state_dict(best_model_state)

    training_loss, training_accuracy = evaluate_model(
        model,
        train_x_tensor,
        train_y_tensor
    )

    return {
        "model": model,
        "training_loss": training_loss,
        "training_accuracy": training_accuracy,
        "validation_loss": best_validation_loss,
        "validation_accuracy": best_validation_accuracy,
        "best_epoch": best_epoch
    }


# ============================================================
# 11. Train and compare all configurations
# ============================================================

results = []

for configuration in configurations:
    print("\nTraining:", configuration["name"])

    result = train_model(configuration)

    result["configuration"] = configuration
    results.append(result)

    print(
        "Architecture:",
        configuration["hidden_sizes"]
    )

    print(
        "Activation:",
        configuration["activation"]
    )

    print(
        "Dropout:",
        configuration["dropout"]
    )

    print(
        "Best epoch:",
        result["best_epoch"]
    )

    print(
        "Training loss:",
        round(result["training_loss"], 6)
    )

    print(
        "Training accuracy:",
        round(result["training_accuracy"], 4)
    )

    print(
        "Validation loss:",
        round(result["validation_loss"], 6)
    )

    print(
        "Validation accuracy:",
        round(result["validation_accuracy"], 4)
    )


# ============================================================
# 12. Sort models by validation loss
#
# The test set is not used to choose the architecture.
# ============================================================

results.sort(
    key=lambda result: result["validation_loss"]
)


print("\n\n========================================")
print("ARCHITECTURE RANKINGS")
print("========================================")

for rank, result in enumerate(results, start=1):
    configuration = result["configuration"]

    print(
        f"{rank}. {configuration['name']}"
    )

    print(
        "   Hidden sizes:",
        configuration["hidden_sizes"]
    )

    print(
        "   Validation loss:",
        round(result["validation_loss"], 6)
    )

    print(
        "   Validation accuracy:",
        round(result["validation_accuracy"], 4)
    )


# ============================================================
# 13. Choose the best model
# ============================================================

best_result = results[0]

best_model = best_result["model"]
best_configuration = best_result["configuration"]


print("\n========================================")
print("BEST CONFIGURATION")
print("========================================")

print("Name:", best_configuration["name"])
print("Hidden sizes:", best_configuration["hidden_sizes"])
print("Activation:", best_configuration["activation"])
print("Dropout:", best_configuration["dropout"])
print("Learning rate:", best_configuration["learning_rate"])
print("Best epoch:", best_result["best_epoch"])


# ============================================================
# 14. Evaluate the best model on the test set
#
# This is the first time the test labels are used.
# ============================================================

test_loss, test_accuracy = evaluate_model(
    best_model,
    test_x_tensor,
    test_y_tensor
)

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# ============================================================
# 15. Show some individual predictions
# ============================================================

best_model.eval()

with torch.no_grad():
    test_logits = best_model(test_x_tensor)

    test_probabilities = torch.sigmoid(test_logits)

    test_predictions = (
        test_probabilities >= 0.5
    ).float()


print("\nFirst 10 predictions:")

for i in range(min(10, len(test_y_tensor))):
    print(
        f"Example {i}: "
        f"probability={test_probabilities[i].item():.4f}, "
        f"prediction={int(test_predictions[i].item())}, "
        f"actual={int(test_y_tensor[i].item())}"
    )