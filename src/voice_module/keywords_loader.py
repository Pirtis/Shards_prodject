import yaml
import os

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMMAND_KEYWORDS = load_yaml(os.path.join(BASE_DIR, "keywords", "commands.yaml"))
SPEC_KEYWORDS = load_yaml(os.path.join(BASE_DIR, "keywords", "specs.yaml"))
