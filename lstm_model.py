import torch
import numpy as np
import pandas as pd
import joblib as jb

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# Importing from out files
from architecture import get_model, elongationCNN, elongationLSTM
from training_testing_functions import train_model, test_model, plot_loss, compute_metrics

# Change this to the model you are training
MODEL_CLASSES = {
    "LSTM": elongationLSTM
}

# Paths that we will use
BASE_DIR = Path(__file__).parent / "creep_data"
processed_data_folder = BASE_DIR / "processed_data"
results_folder = BASE_DIR / "results" / "LSTM"

# Hyperparams that are common throughout all the models
seed = 42
epochs = 100
learning_rate = 0.001
batch_size = 32
early_stopping = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
np.random.seed(seed=seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_tensor(name):
    return torch.load(processed_data_folder / f"{name}.pt")

def build_loader(X, Y, batch_size, shuffle):
    dataset = TensorDataset(X, Y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    results_folder.mkdir(parents=True, exist_ok=True)

    # Loading processed tensors
    X_train = load_tensor("X_train")
    Y_train = load_tensor("y_train")
    X_val = load_tensor("X_val")
    Y_val = load_tensor("y_val")
    X_test = load_tensor("X_test")
    Y_test = load_tensor("y_test")

    test_case_ids = torch.load(processed_data_folder / "test_sequence_case_ids.pt")

    # Loading fitted scaler
    scaler = jb.load(processed_data_folder / "scaler.joblib")
    features_to_scale = jb.load(processed_data_folder / "features_to_scale.joblib")

    input_size = X_train.shape[2]
    seq_len = X_train.shape[1]

    # Building DataLoaders
    train_loader = build_loader(X_train, Y_train, batch_size, shuffle=True)
    val_loader = build_loader(X_val, Y_val, batch_size, shuffle=False)
    test_loader = build_loader(X_test, Y_test, batch_size, shuffle=False)

    results_summary = []

    # Training and evaluate CNN and LSTM models
    for model_name in MODEL_CLASSES:
        print(f"\n{'=' * 25}    {model_name}    {'=' * 25}")

        model = get_model(model_name, input_size=input_size, seq_len=seq_len)
        num_params = sum(p.numel() for p in model.parameters())

        model, history = train_model(model, train_loader, val_loader, num_epochs=epochs, lr=learning_rate, device=device, early_stopping_patience=early_stopping)
        preds_scaled, targets_scaled, case_ids = test_model(model, test_loader, test_case_ids=test_case_ids, device=device)
        metrics, preds_real, targets_real = compute_metrics(preds_scaled, targets_scaled, scaler=scaler, features_to_scale=features_to_scale, print_results=True)

        results_summary.append({
            "model": model_name,
            "rmse_mm": metrics["RMSE"],
            "mae_mm": metrics["MAE"],
            "R2": metrics["R2"],
            "num_params": num_params,
            "epochs_trained": len(history["train_loss"]),
        })

        plot_loss(history, save_path = results_folder / f'{model_name}_loss_curve.png')
        torch.save(model.state_dict(), results_folder / f'{model_name}_weights.pt')

    # Saving results; Change the file name to match the model you are training
    results_df = pd.DataFrame(results_summary).sort_values("rmse_mm")
    results_df.to_csv(results_folder / "lstm_model_comparison.csv", index=False)

    print(f"\n{'=' * 50}")
    print("Final results (sorted by RMSE):")
    print(results_df.to_string(index=False))
    print(f"\nSaved to: {results_folder.resolve()}")

if __name__ == "__main__":
    main()