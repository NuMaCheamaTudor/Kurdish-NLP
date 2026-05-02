import re


def clean_text(text):
    text = text.strip()
    if len(text) < 10:
        return None
    if any(char.isdigit() for char in text):
        return None
    return text


input_path = "data/raw/asosoft_small.txt"
output_path = "data/processed/asosoft_clean.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    line = clean_text(line)
    if line:
        cleaned.append(line)

with open(output_path, "w", encoding="utf-8") as f:
    for line in cleaned[:100000]:  # limit
        f.write(line + "\n")

print("Cleaned corpus saved.")
