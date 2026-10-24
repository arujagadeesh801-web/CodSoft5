import argparse

import matplotlib.pyplot as plt

from data_preprocessing import (
    MODEL_DIR,
    OUTPUT_DIR,
    create_vocabulary,
    load_text,
    prepare_training_data,
    save_vocabulary,
)
from model import build_model


SEQUENCE_LENGTH = 80
MODEL_PATH = MODEL_DIR / "handwritten_rnn.keras"
LOSS_PLOT_PATH = OUTPUT_DIR / "training_loss.png"


def train(epochs=30, batch_size=64):
    """Train the character-level language model and save its artifacts."""
    if epochs < 1:
        raise ValueError("Epoch count must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    text = load_text()
    char_to_index, index_to_char = create_vocabulary(text)
    inputs, targets = prepare_training_data(
        text,
        char_to_index,
        sequence_length=SEQUENCE_LENGTH,
    )

    model = build_model(
        vocab_size=len(char_to_index),
        sequence_length=SEQUENCE_LENGTH,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on {len(text):,} characters "
        f"({len(inputs):,} sequences, {len(char_to_index):,} unique characters)."
    )
    history = model.fit(
        inputs,
        targets,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
    )

    model.save(MODEL_PATH)
    save_vocabulary(char_to_index, index_to_char)

    plt.figure()
    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    print(f"Trained model saved to: {MODEL_PATH}")
    print(f"Training loss graph saved to: {LOSS_PLOT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the character-level handwritten text generator."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()