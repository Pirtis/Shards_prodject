from keywords_loader import COMMAND_KEYWORDS
from ml_classifier import classify_with_model

def detect_command(text: str):
    text = text.lower()
    for command, phrases in COMMAND_KEYWORDS.items():
        for phrase in phrases:
            if phrase in text:
                return command
    return classify_with_model(text)
