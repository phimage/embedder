import subprocess
import numpy as np
import os

class MyCmdEmbedder:
    def encode(self, sentences, batch_size=32, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = []
        for sentence in sentences:
            try:
                
                # Set up environment with the model path
                env = os.environ.copy()
                env['EMBEDDING_MODEL_PATH'] = os.path.expanduser('~/.cache/huggingface/hub/models--Xenova--nomic-embed-text-v1/snapshots/0b85f78966a655763985a595b770f221374dda10')

                # Call the command line tool and get output
                result = subprocess.run(
                    ['../embedder', sentence],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding='utf-8',
                    env=env,
                    check=True
                )
                # Parse the output into a list of floats
                emb = [float(x) for x in result.stdout.strip().split()]
                
                # Validate embedding quality
                if len(emb) == 0:
                    raise ValueError("Empty embedding returned")
                if any(np.isnan(emb)) or any(np.isinf(emb)):
                    raise ValueError("Embedding contains NaN or Inf values")
                
                embeddings.append(emb)
            except subprocess.CalledProcessError as e:
                print(f"Error running embedder: {e}")
                print(f"stderr: {e.stderr}")
                raise
            except ValueError as e:
                print(f"Error processing embedding: {e}")
                raise
        return embeddings