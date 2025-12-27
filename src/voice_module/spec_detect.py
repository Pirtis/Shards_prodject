import re
from keywords_loader import SPEC_KEYWORDS

def detect_spec(text: str) -> str:
    words = re.findall(r"[а-яё]+", text.lower())
    for word in words:
        for spec, variants in SPEC_KEYWORDS.items():
            if word in variants:
                return spec
    return "all"

