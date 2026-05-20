"""Load a trained model and make predictions on new data."""
import torch
import json
import argparse
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
    
    # Load normalization statistics from model metadata
    checkpoint = torch.load(args.model, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    Y_mean = metadata.get("Y_mean")
    Y_std = metadata.get("Y_std")

    # Load data
    with open(args.data, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    X = data["X"]
    Y = data["Y"]
    if (Y_mean is None or Y_std is None) and ("Y_mean" in data and "Y_std" in data):
        Y_mean = data.get("Y_mean")
        Y_std = data.get("Y_std")

    if args.sample >= len(X):
        print(f"Error: sample index {args.sample} out of range (max {len(X)-1})")
        return

    expected_input_dim = 3
    expected_output_dim = 1

    # Make prediction
    sample_input = torch.tensor(X[args.sample], dtype=torch.float32).unsqueeze(0)
    actual_output = torch.tensor(Y[args.sample], dtype=torch.float32).unsqueeze(0)

    if sample_input.shape[1] != expected_input_dim:
        print(
            f"Error: sample width {sample_input.shape[1]} does not match expected input_dim {expected_input_dim}"
        )
        return

    model_input_dim = model[0].in_features
    if sample_input.shape[1] != model_input_dim:
        print(
            f"Error: sample width {sample_input.shape[1]} does not match model input_dim {model_input_dim}"
        )
        return
    model_output_dim = model[2].out_features
    if actual_output.shape[1] != expected_output_dim:
        print(
            f"Error: sample label width {actual_output.shape[1]} does not match expected output_dim {expected_output_dim}"
        )
        return
    if model_output_dim != expected_output_dim:
        print(
            f"Error: model output_dim {model_output_dim} does not match expected output_dim {expected_output_dim}"
        )
        return

    with torch.no_grad():  # Disable gradient computation for inference
        predicted_output = model(sample_input)
    
    # Denormalize predictions if statistics are available
    predicted_denorm = predicted_output.clone()
    actual_denorm = actual_output.clone()
    if Y_mean is not None and Y_std is not None:
        Y_mean_tensor = torch.tensor(Y_mean, dtype=torch.float32).unsqueeze(0)
        Y_std_tensor = torch.tensor(Y_std, dtype=torch.float32).unsqueeze(0)
        predicted_denorm = predicted_output * Y_std_tensor + Y_mean_tensor
        actual_denorm = actual_output * Y_std_tensor + Y_mean_tensor

    # Display results
    print(f"\n--- Sample {args.sample} ---")
    print(f"Predicted deltas (normalized): {predicted_output.squeeze().tolist()}")
    print(f"Predicted deltas (original):   {predicted_denorm.squeeze().tolist()}")
    print(f"Actual deltas (normalized):    {actual_output.squeeze().tolist()}")
    print(f"Actual deltas (original):      {actual_denorm.squeeze().tolist()}")

    # Calculate error on normalized data
    mse = torch.nn.functional.mse_loss(predicted_output, actual_output)
    mae = torch.nn.functional.l1_loss(predicted_output, actual_output)
    print(f"\nNormalized - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Calculate error on original scale
    mse_denorm = torch.nn.functional.mse_loss(predicted_denorm, actual_denorm)
    mae_denorm = torch.nn.functional.l1_loss(predicted_denorm, actual_denorm)
    print(f"Original Scale - MSE: {mse_denorm:.4f}, MAE: {mae_denorm:.4f}")

    # Batch inference
    print(f"\n--- Batch inference on first 10 samples ---")
    batch_size = min(10, len(X))
    batch_input = torch.tensor([X[i] for i in range(batch_size)], dtype=torch.float32)
    batch_actual = torch.tensor([Y[i] for i in range(batch_size)], dtype=torch.float32)

    if batch_input.shape[1] != expected_input_dim:
        print(
            f"Error: batch width {batch_input.shape[1]} does not match expected input_dim {expected_input_dim}"
        )
        return

    if batch_input.shape[1] != model_input_dim:
        print(
            f"Error: batch width {batch_input.shape[1]} does not match model input_dim {model_input_dim}"
        )
        return

    with torch.no_grad():
        batch_pred = model(batch_input)
    
    # Denormalize batch predictions if statistics are available
    batch_pred_denorm = batch_pred.clone()
    batch_actual_denorm = batch_actual.clone()
    if Y_mean is not None and Y_std is not None:
        Y_mean_tensor = torch.tensor(Y_mean, dtype=torch.float32).unsqueeze(0)
        Y_std_tensor = torch.tensor(Y_std, dtype=torch.float32).unsqueeze(0)
        batch_pred_denorm = batch_pred * Y_std_tensor + Y_mean_tensor
        batch_actual_denorm = batch_actual * Y_std_tensor + Y_mean_tensor

    batch_mse = torch.nn.functional.mse_loss(batch_pred, batch_actual)
    batch_mae = torch.nn.functional.l1_loss(batch_pred, batch_actual)
    print(f"Batch Normalized - MSE: {batch_mse:.4f}, MAE: {batch_mae:.4f}")
    
    batch_mse_denorm = torch.nn.functional.mse_loss(batch_pred_denorm, batch_actual_denorm)
    batch_mae_denorm = torch.nn.functional.l1_loss(batch_pred_denorm, batch_actual_denorm)
    print(f"Batch Original Scale - MSE: {batch_mse_denorm:.4f}, MAE: {batch_mae_denorm:.4f}")


if __name__ == "__main__":
    main()
