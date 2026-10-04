import unicodedata
text = """In 2026, we teach “intro-to-AI” with hands-on labs—no hype. Students ask: “Why tokens?”
Because models read pieces, not words. E.g., ‘ChatGPT-5’ ≠ ‘Chat’, ‘GPT’, ‘5’ in all schemes.
We track loss/accuracy, compare char/word/BPE, and test a URL: https://example.org/a/b?c=42.
Café prices rose 3.7%—blame supply-chain weirdness (and ☕ demand)."""

# Convert to lowercase
clean_text = text.lower()

# Replace punctuation with whitespace
clean_text = "".join(
    " " if unicodedata.category(character).startswith("P") else character
    for character in clean_text
)
print("Cleaned text:", clean_text)
print()

# tokenize the text by splitting on whitespace
tokens = clean_text.split()
print("Tokens:", tokens)
print()
tokenized_length = len(tokens)
print("Tokenized length:", tokenized_length)
print()

# Build a sorted vocabulary of unique tokens
vocab = sorted(set(tokens))
print("Vocabulary:", vocab)
print()

# Assign a unique numerical ID to each word in vocab (encoding)
word_to_id = {}
for word_id, word in enumerate(vocab):
    word_to_id[word] = word_id
print("Numerical IDs for Words:", word_to_id)
print()

# Map each token to its numerical ID
token_ids = [word_to_id[word] for word in tokens]
print("Token IDs:", token_ids)
print()

# Reverse the mapping: numerical ID -> word (decoding logic)
id_to_word = {word_id: word for word, word_id in word_to_id.items()}
print("Decoded ID to word:", id_to_word)
print()

# Decode the token IDs back to words
decoded_tokens = []
for token_id in token_ids:
    decoded_tokens.append(id_to_word[token_id])

# Join the decoded words back into text
decoded_text = " ".join(decoded_tokens)
print("Decoded text from token IDs:", decoded_text)