#!/usr/bin/env python3
import os
from transformers import AutoTokenizer

# Test tokenization
model_path = os.path.expanduser('~/.cache/huggingface/hub/models--Xenova--nomic-embed-text-v1/snapshots/0b85f78966a655763985a595b770f221374dda10')
tokenizer = AutoTokenizer.from_pretrained(model_path)

test_text = "Hello world"
tokens = tokenizer.tokenize(test_text)
token_ids = tokenizer.encode(test_text, add_special_tokens=True)

print(f"Text: '{test_text}'")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

# Test some other examples
test_sentences = [
    "This is a test",
    "BERT tokenization with WordPiece",
    "Machine learning is fascinating"
]

for sentence in test_sentences:
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    print(f"\nText: '{sentence}'")
    print(f"Token IDs: {token_ids}")
