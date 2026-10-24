import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

TEXT_FILE = DATASET_DIR / "text_corpus.txt"
VOCAB_FILE = MODEL_DIR / "vocabulary.json"


def load_text():
    """Load the training text from the dataset folder."""

    if not TEXT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {TEXT_FILE}\n"
            "Place your text corpus in dataset/text_corpus.txt"
        )

    text = TEXT_FILE.read_text(encoding="utf-8")

    # Remove unnecessary leading and trailing whitespace.
    text = text.strip()

    if len(text) < 100:
        raise ValueError(
            "The training corpus is too short. "
            "Please provide a larger text dataset."
        )

    return text


def create_vocabulary(text):
    """Create character-to-index and index-to-character mappings."""

    characters = sorted(set(text))

    char_to_index = {
        character: index
        for index, character in enumerate(characters)
    }

    index_to_char = {
        str(index): character
        for index, character in enumerate(characters)
    }

    return char_to_index, index_to_char


def encode_text(text, char_to_index):
    """Convert text characters into integer IDs."""

    return np.array(
        [char_to_index[character] for character in text],
        dtype=np.int32
    )


def save_vocabulary(char_to_index, index_to_char):
    """Save vocabulary mappings for future text generation."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    vocabulary = {
        "char_to_index": char_to_index,
        "index_to_char": index_to_char
    }

    with VOCAB_FILE.open("w", encoding="utf-8") as file:
        json.dump(vocabulary, file, ensure_ascii=False, indent=4)

    print(f"Vocabulary saved to: {VOCAB_FILE}")


def load_vocabulary():
    """Load the saved character vocabulary."""

    if not VOCAB_FILE.exists():
        raise FileNotFoundError(
            "Vocabulary not found. Run src/train.py first."
        )

    with VOCAB_FILE.open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    char_to_index = vocabulary["char_to_index"]

    index_to_char = {
        int(index): character
        for index, character in vocabulary["index_to_char"].items()
    }

    return char_to_index, index_to_char


def prepare_training_data(text, char_to_index, sequence_length=80):
    """
    Create input sequences and their next-character targets.

    Example:
        Input:  hell
        Target: ello
    """

    encoded_text = encode_text(text, char_to_index)

    if len(encoded_text) <= sequence_length:
        raise ValueError(
            f"Training text must contain more than "
            f"{sequence_length} characters."
        )

    inputs = []
    targets = []

    for start in range(
        0,
        len(encoded_text) - sequence_length,
        sequence_length
    ):
        sequence = encoded_text[start:start + sequence_length + 1]

        inputs.append(sequence[:-1])
        targets.append(sequence[1:])

    X = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)

    if len(X) == 0:
        raise ValueError("Could not create training sequences.")

    return X, y