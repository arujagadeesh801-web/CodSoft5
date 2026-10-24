# Handwritten Text Generation Using Character-Level RNN

## Project Overview

This project implements a character-level recurrent neural network using LSTM to learn patterns from a text corpus and generate new text.

The generated text can also be rendered as a handwriting-style image using a font-based rendering module.

## Objectives

- Understand character-level text generation.
- Preprocess text data for neural network training.
- Implement an LSTM-based recurrent neural network.
- Train a model to predict the next character.
- Generate new text using the trained model.
- Visualize training loss.
- Convert generated text into a handwriting-style image.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Pillow
- LSTM (Long Short-Term Memory)

## Project Structure

```text
Handwritten_Text_Generation/
├── dataset/
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── handwriting_renderer.py
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset

Place the prepared text corpus at:

`dataset/text_corpus.txt`

The dataset must contain enough readable text for character-level language modeling.

If the original dataset contains handwritten images, appropriate preprocessing or transcription extraction is required before using this text-based implementation.

## Training

Run:

```bash
python src/train.py
```

Training requires a UTF-8 text corpus at `dataset/text_corpus.txt` (at least
100 characters). To change the training duration or batch size, for example:

```bash
python src/train.py --epochs 50 --batch-size 32
```

The trained model and vocabulary will be saved in the `models/` directory.

The training loss graph will be saved as `outputs/training_loss.png`.

## Generate Text

Run:

```bash
python src/generate.py
```

Enter a starting prompt, generation length, and temperature when prompted.

Generated text will be saved as `outputs/generated_text.txt`.

## Generate a Handwriting-Style Image

Run:

```bash
python src/handwriting_renderer.py
```

The image will be saved as `outputs/handwritten_output.png`.

The rendering module uses fonts and does not generate realistic handwriting strokes with a neural network.

## Expected Results

- Trained LSTM model
- Character vocabulary
- Generated text
- Training loss graph
- Handwriting-style PNG image

## Limitations

- Output quality depends on the training corpus.
- Small datasets may produce repetitive or incoherent text.
- The text-generation model does not directly generate handwriting images.
- Font-based rendering does not reproduce individual handwriting styles.

## Conclusion

This project demonstrates how a character-level LSTM can learn character sequences and generate new text. It also includes a separate rendering step to visualize generated text in a handwriting-style format."# CodSoft5" 
