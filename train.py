import torch
import torch.nn as nn
import json
import argparse
import time
from pathlib import Path


EXPECTED_INPUT_DIM = 3
EXPECTED_OUTPUT_DIM = 1


def save_model(model: nn.Module, path: str = "model.pth", metadata: dict = None):
    """Save model state_dict and optional metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {}
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str = "model.pth") -> nn.Module:
    """Load model from checkpoint."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    
    checkpoint = torch.load(path, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    input_dim = metadata.get("input_dim")
    if input_dim is None:
        input_dim = checkpoint["model_state_dict"]["0.weight"].shape[1]
    
    output_dim = metadata.get("output_dim", EXPECTED_OUTPUT_DIM)

    # reconstruct model (must match architecture in main)
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {path} (input_dim={input_dim}, output_dim={output_dim})")
    return model


def create_model(input_dim: int) -> nn.Module:
    """Create a fresh model."""
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, EXPECTED_OUTPUT_DIM)
    )


def main():
    parser = argparse.ArgumentParser(description="Train delta position predictor")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model checkpoint")
    parser.add_argument("--save-model", type=str, default="model.pth", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N seconds")
    args = parser.parse_args()

    with open("preprocessed_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    X = data["X"]
    Y = data["Y"]
    # Load normalization statistics
    X_mean = data.get("X_mean")
    X_std = data.get("X_std")
    Y_mean = data.get("Y_mean")
    Y_std = data.get("Y_std")

    if not X:
        raise ValueError("No training samples found in preprocessed_data.json")

    input_dim = len(X[0])
    if input_dim != EXPECTED_INPUT_DIM:
        raise ValueError(
            f"Expected input_dim={EXPECTED_INPUT_DIM}, but preprocessed_data.json contains input_dim={input_dim}. Run process_data.py again with single-joint 3-feature format."
        )
    output_dim = len(Y[0])
    if output_dim != EXPECTED_OUTPUT_DIM:
        raise ValueError(
            f"Expected output_dim={EXPECTED_OUTPUT_DIM}, but preprocessed_data.json contains output_dim={output_dim}."
        )
    if any(len(sample) != input_dim for sample in X):
        raise ValueError("Inconsistent input widths found in X")
    if any(len(sample) != output_dim for sample in Y):
        raise ValueError("Inconsistent output widths found in Y")

    # Load or create model
    if args.load_model:
        model = load_model(args.load_model)
        expected_input_dim = model[0].in_features
        if expected_input_dim != input_dim:
            raise ValueError(
                f"Checkpoint input_dim={expected_input_dim} does not match data input_dim={input_dim}"
            )
        expected_output_dim = model[2].out_features
        if expected_output_dim != output_dim:
            raise ValueError(
                f"Checkpoint output_dim={expected_output_dim} does not match data output_dim={output_dim}"
            )
    else:
        model = create_model(input_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    epochs = args.epochs

    print(model)
    print(f"Model dimensions: input_dim={model[0].in_features}, output_dim={model[2].out_features}")

    sample = X[0]
    sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # add batch dimension

    print("sample.shape:", sample.shape)

    last_save_time = time.time()
    last_checkpoint = None
    save_interval_secs = args.save_interval

    for epoch in range(epochs):
        print("-" * 30)
        print(f"Training Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        for i in range(len(X)):
            sample = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)  # add batch dimension
            target = torch.tensor(Y[i], dtype=torch.float32).unsqueeze(0)

            output = model(sample)
            loss = loss_fn(output, target)
            total_loss += loss.item()

            optimizer.zero_grad() #clear past gradients
            loss.backward() #compute gradients
            optimizer.step() #update weights

        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        print("-" * 30)
        
        # Save checkpoint every N seconds
        current_time = time.time()
        if current_time - last_save_time >= save_interval_secs:
            # Delete previous checkpoint if it exists
            if last_checkpoint and Path(last_checkpoint).exists():
                Path(last_checkpoint).unlink()
                print(f"Deleted previous checkpoint: {last_checkpoint}")
            
            # Save new checkpoint
            ckpt_path = str(Path(args.save_model).stem) + f"_epoch{epoch+1}.pth"
            save_model(model, ckpt_path, {
                "epoch": epoch+1,
                "loss": avg_loss,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "X_mean": X_mean,
                "X_std": X_std,
                "Y_mean": Y_mean,
                "Y_std": Y_std
            })
            last_checkpoint = ckpt_path
            last_save_time = current_time

    # Save final model
    save_model(model, args.save_model, {
        "epochs_trained": epochs,
        "final_loss": avg_loss,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std
    })
    
    # Clean up all intermediate checkpoints
    print("\nCleaning up intermediate checkpoints...")
    model_stem = Path(args.save_model).stem
    for ckpt in Path(".").glob(f"{model_stem}_epoch*.pth"):
        ckpt.unlink()
        print(f"Deleted: {ckpt}")

    sample = torch.tensor(X[0], dtype=torch.float32).unsqueeze(0)
    target = torch.tensor(Y[0], dtype=torch.float32).unsqueeze(0)

    prediction = model(sample)

    print("Prediction:")
    print(prediction)

    print("Actual:")
    print(target)

    loss = loss_fn(prediction, target)
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()
