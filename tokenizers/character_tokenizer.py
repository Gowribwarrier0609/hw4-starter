text = """In 2026, we teach “intro-to-AI” with hands-on labs—no hype. Students ask: “Why tokens?”
Because models read pieces, not words. E.g., ‘ChatGPT-5’ ≠ ‘Chat’, ‘GPT’, ‘5’ in all schemes.
We track loss/accuracy, compare char/word/BPE, and test a URL: https://example.org/a/b?c=42.
Café prices rose 3.7%—blame supply-chain weirdness (and ☕ demand)."""

# Build the character vocabulary
chars = sorted(list(set(text)))

# Convert character -> number ID
stoi = {c: i for i, c in enumerate(chars)}

# Convert number ID -> character
itos = {i: c for c, i in stoi.items()}

# Convert the full text into character token IDs
tokens = [stoi[c] for c in text]

# Print useful information
print("Corpus length:", len(text))
print("Vocab:", chars)
print("Vocab size:", len(chars))
print("Tokenized length:", len(tokens))