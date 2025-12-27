from config import OUTPUT_FILE

def save(command: str, spec: str):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(command + "\n")
        f.write(spec + "\n")
