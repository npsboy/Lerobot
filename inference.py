"""Load a trained model and make predictions on new data."""
import torch
import json
import argparse
from pathlib import Path
from train import load_model


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="preprocessed_data.json", help="Path to preprocessed data")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to predict")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    # Load data
    with open(args.data, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    X = data["X"]
    Y = data["Y"]

    if args.sample >= len(X):
        print(f"Error: sample index {args.sample} out of range (max {len(X)-1})")
        return

    # Make prediction
    sample_input = torch.tensor(X[args.sample], dtype=torch.float32).unsqueeze(0)
    actual_output = torch.tensor(Y[args.sample], dtype=torch.float32).unsqueeze(0)

    expected_input_dim = model[0].in_features
    if sample_input.shape[1] != expected_input_dim:
        print(
            f"Error: sample width {sample_input.shape[1]} does not match model input_dim {expected_input_dim}"
        )
        return

    with torch.no_grad():  # Disable gradient computation for inference
        predicted_output = model(sample_input)

    # Display results
    print(f"\n--- Sample {args.sample} ---")
    print(f"Predicted delta positions: {predicted_output.squeeze().tolist()}")
    print(f"Actual delta positions:    {actual_output.squeeze().tolist()}")

    # Calculate error
    mse = torch.nn.functional.mse_loss(predicted_output, actual_output)
    mae = torch.nn.functional.l1_loss(predicted_output, actual_output)
    print(f"\nMSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Batch inference
    print(f"\n--- Batch inference on first 10 samples ---")
    batch_size = min(10, len(X))
    batch_input = torch.tensor([X[i] for i in range(batch_size)], dtype=torch.float32)
    batch_actual = torch.tensor([Y[i] for i in range(batch_size)], dtype=torch.float32)

    if batch_input.shape[1] != expected_input_dim:
        print(
            f"Error: batch width {batch_input.shape[1]} does not match model input_dim {expected_input_dim}"
        )
        return

    with torch.no_grad():
        batch_pred = model(batch_input)

    batch_mse = torch.nn.functional.mse_loss(batch_pred, batch_actual)
    batch_mae = torch.nn.functional.l1_loss(batch_pred, batch_actual)
    print(f"Batch MSE: {batch_mse:.4f}")
    print(f"Batch MAE: {batch_mae:.4f}")


if __name__ == "__main__":
    main()
