# import warnings
# warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
# warnings.filterwarnings("ignore", message="overflow encountered in matmul")
# warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

import os
# Set optimal thread count for Apple Silicon
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

from my_sentence_transformer import MySentenceTransformer
model = MySentenceTransformer()

#  just test before running the MTEB benchmark
sentences = ["Hello world!", "This is a test."]
embeddings = model.encode(sentences)
print(f"Generated embeddings shape: {len(embeddings)}x{len(embeddings[0])}")
print("First few values of first embedding:", embeddings[0][:10])

# https://huggingface.co/blog/mteb#benchmark-your-model
from mteb import MTEB
 
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/MySentenceTransformer")
