import re


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", "")

    # Replace weird space characters or non-breaking spaces with standard space
    text = text.replace("\xa0", " ")

    # Standardize newline characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multi-blank lines down to max 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple inline spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
