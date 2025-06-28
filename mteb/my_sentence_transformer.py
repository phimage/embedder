from sentence_transformers import SentenceTransformer
from my_embedder_wrapper import MyCmdEmbedder

class MySentenceTransformer(SentenceTransformer):
    def __init__(self):
        super().__init__(modules=[])
        self._cmd_embedder = MyCmdEmbedder()

    def encode(self, sentences, **kwargs):
        return self._cmd_embedder.encode(sentences, **kwargs)