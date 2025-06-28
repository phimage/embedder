# import warnings
# warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
# warnings.filterwarnings("ignore", message="overflow encountered in matmul")
# warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

from my_sentence_transformer import MySentenceTransformer
model = MySentenceTransformer()

#  just test before running the MTEB benchmark
sentences = ["Hello world!", "This is a test."]
embeddings = model.encode(sentences)
print(embeddings)

# https://huggingface.co/blog/mteb#benchmark-your-model
from mteb import MTEB
 
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/MySentenceTransformer")