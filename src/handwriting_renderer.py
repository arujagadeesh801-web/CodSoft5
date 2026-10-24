from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from data_preprocessing import OUTPUT_DIR


def load_font(font_size=30):
    """
    Try common fonts that may be installed on the system.

    For a more handwritten appearance, provide a handwriting
    font file and set HANDWRITING_FONT_PATH to its location.
    """

    font_path = __import__("os").environ.get(
        "HANDWRITING_FONT_PATH"
    )

    candidates = []

    if font_path:
        candidates.append(font_path)

    candidates.extend([
        "C:/Windows/Fonts/segoepr.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])

    for candidate in candidates:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, font_size)
            except OSError:
                continue

    # Fallback font; may not look handwritten.
    return ImageFont.load_default()


def render_handwriting(
    text,
    output_path=None,
    width=1200,
    padding=60,
    font_size=30,
    line_spacing=16
):
    """Render generated text into a PNG image."""

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    font = load_font(font_size)

    # Create an initial image to measure text.
    temporary_image = Image.new("RGB", (width, 100), "white")
    temporary_draw = ImageDraw.Draw(temporary_image)

    max_text_width = width - 2 * padding

    lines = []

    for paragraph in text.splitlines() or [text]:
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split()
        current_line = ""

        for word in words:
            candidate = (
                f"{current_line} {word}".strip()
            )

            candidate_width = temporary_draw.textbbox(
                (0, 0),
                candidate,
                font=font
            )[2]

            if candidate_width <= max_text_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)

                current_line = word

        if current_line:
            lines.append(current_line)

    line_height = font_size + line_spacing

    image_height = (
        padding * 2
        + max(1, len(lines)) * line_height
    )

    image = Image.new(
        "RGB",
        (width, image_height),
        "#fffdf5"
    )

    draw = ImageDraw.Draw(image)

    y = padding

    for line in lines:
        draw.text(
            (padding, y),
            line,
            fill="#202020",
            font=font
        )

        y += line_height

    if output_path is None:
        output_path = OUTPUT_DIR / "handwritten_output.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(output_path)

    print(f"Handwriting-style image saved to: {output_path}")

    return output_path


def main():
    text_path = OUTPUT_DIR / "generated_text.txt"

    if not text_path.exists():
        raise FileNotFoundError(
            "Generated text not found. Run src/generate.py first."
        )

    text = text_path.read_text(encoding="utf-8")

    render_handwriting(text)


if __name__ == "__main__":
    main()