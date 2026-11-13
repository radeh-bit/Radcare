import base64
import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic


def read_image_as_base64(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {file_path}")

    with path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_invoice(image_path: str):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY ist nicht gesetzt. Bitte export ANTHROPIC_API_KEY=...")

    client = Anthropic(api_key=api_key)
    image_data = read_image_as_base64(image_path)

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Lies diese Rechnung. Antworte NUR mit JSON im Format: "
                            "{\"lieferant\": \"\", \"datum\": \"YYYY-MM-DD\", "
                            "\"positionen\": [{\"name\": \"\", \"menge\": 1, \"einzelpreis\": 12.34}], "
                            "\"summe\": 0.0, \"mwst\": 0.0}. "
                            "Wenn die Rechnung mehrere Positionen hat, liste sie alle auf. "
                            "Verwende Zahlen als Zahlen, nicht als Strings."
                        ),
                    },
                ],
            }
        ],
    )

    text = response.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Anthropic hat kein gültiges JSON zurückgegeben: {text}") from exc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python invoice_parser.py <rechnung.jpg>")
        sys.exit(1)

    result = parse_invoice(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
