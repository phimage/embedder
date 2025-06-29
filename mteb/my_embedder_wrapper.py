import subprocess
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class MyCmdEmbedder:
    def __init__(self):
        # Set up environment once
        self.env = os.environ.copy()
        self.env['EMBEDDING_MODEL_PATH'] = os.path.expanduser('~/.cache/huggingface/hub/models--Xenova--nomic-embed-text-v1/snapshots/0b85f78966a655763985a595b770f221374dda10')
        # Optimize for Apple Silicon
        self.env['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
        
    def _process_single_sentence(self, sentence):
        """Process a single sentence through the embedder"""
        try:
            # Call the command line tool and get output
            result = subprocess.run(
                ['../embedder', sentence],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                env=self.env,
                check=True
            )
            # Parse the output into a list of floats
            emb = [float(x) for x in result.stdout.strip().split()]
            
            # Validate embedding quality
            if len(emb) == 0:
                raise ValueError("Empty embedding returned")
            if any(np.isnan(emb)) or any(np.isinf(emb)):
                raise ValueError("Embedding contains NaN or Inf values")
                
            return emb
        except subprocess.CalledProcessError as e:
            print(f"Error running embedder: {e}")
            print(f"stderr: {e.stderr}")
            raise
        except ValueError as e:
            print(f"Error processing embedding: {e}")
            raise
    
    def encode(self, sentences, batch_size=32, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # For small batches, use serial processing to avoid overhead
        if len(sentences) <= 4:
            embeddings = []
            for sentence in sentences:
                emb = self._process_single_sentence(sentence)
                embeddings.append(emb)
            return embeddings
        
        # For larger batches, use parallel processing
        max_workers = min(4, multiprocessing.cpu_count())  # Limit concurrent processes
        embeddings = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process sentences in parallel
            future_to_sentence = {
                executor.submit(self._process_single_sentence, sentence): i 
                for i, sentence in enumerate(sentences)
            }
            
            # Collect results in order
            results = [None] * len(sentences)
            for future in future_to_sentence:
                try:
                    idx = future_to_sentence[future]
                    results[idx] = future.result()
                except Exception as exc:
                    print(f'Sentence generated an exception: {exc}')
                    raise
            
            embeddings = results
        
        return embeddings