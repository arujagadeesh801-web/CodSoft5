from pathlib import Path

import numpy as np
import tensorflow as tf

from data_preprocessing import load_vocabulary, MODEL_DIR, OUTPUT_DIR


SEQUENCE_LENGTH = 80


def generate_text(
    model,
    prompt,
    char_to_index,
    index_to_char,
    length=500,
    temperature=0.8
):
    """
    Generate text one character at a time.

    Higher temperature:
        More random output.

    Lower temperature:
        More predictable output.
    """

    if not 0 < temperature:
        raise ValueError("Temperature must be greater than zero.")

    if length < 1:
        raise ValueError("Length must be at least 1.")

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    # Replace unsupported characters with a known character.
    fallback_char = next(iter(char_to_index))

    prompt = "".join(
        character if character in char_to_index else fallback_char
        for character in prompt
    )

    generated = list(prompt)

    for _ in range(length):
        context = generated[-SEQUENCE_LENGTH:]

        # Pad short prompts to match the model's input shape.
        if len(context) < SEQUENCE_LENGTH:
            padding_char = " " if " " in char_to_index else fallback_char

            context = (
                [padding_char] * (SEQUENCE_LENGTH - len(context))
                + context
            )

        input_ids = np.array(
            [[char_to_index[character] for character in context]],
            dtype=np.int32
        )

        # Predict the next character.
        predictions = model(
            input_ids,
            training=False
        ).numpy()[0, -1, :]

        # Temperature controls prediction randomness.
        predictions = predictions / temperature

        # Stable softmax.
        predictions = predictions - np.max(predictions)

        probabilities = np.exp(predictions)
        probabilities = probabilities / probabilities.sum()

        next_id = np.random.choice(
            len(probabilities),
            p=probabilities
        )

        next_character = index_to_char[int(next_id)]

        generated.append(next_character)

    return "".join(generated)


def main():
    model_path = MODEL_DIR / "handwritten_rnn.keras"
    output_path = OUTPUT_DIR / "generated_text.txt"

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train.py first."
        )

    print("Loading trained model...")

    model = tf.keras.models.load_model(model_path)

    char_to_index, index_to_char = load_vocabulary()

    print("\nCharacter-Level RNN Text Generator")
    print("----------------------------------")

    prompt = input(
        "Enter starting text (example: The): "
    ).strip()

    if not prompt:
        prompt = "The"

    try:
        length = int(
            input("Number of new characters (default 500): ") or "500"
        )

        temperature = float(
            input("Temperature (default 0.8): ") or "0.8"
        )

        generated_text = generate_text(
            model=model,
            prompt=prompt,
            char_to_index=char_to_index,
            index_to_char=index_to_char,
            length=length,
            temperature=temperature
        )

    except ValueError as error:
        print(f"Invalid input: {error}")
        return

    print("\n" + "=" * 50)
    print("GENERATED TEXT")
    print("=" * 50)
    print(generated_text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        generated_text,
        encoding="utf-8"
    )

    print(f"\nGenerated text saved to: {output_path}")


if __name__ == "__main__":
    main()