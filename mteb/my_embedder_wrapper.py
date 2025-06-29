import subprocess
import numpy as np
import os

class MyCmdEmbedder:
    """
    Improved embedder wrapper using efficient batch processing.
    
    Key improvements over the previous version:
    - Uses native C++ batch processing instead of Python threading
    - Single subprocess call for multiple sentences (reduced overhead)
    - Safe handling of texts containing newlines via null-delimiter
    - Better memory utilization with batch tensor processing
    - Significantly faster for multiple sentences
    - Eliminates complex thread pool management
    """
    def __init__(self, batch_size=64, embedder_path='../embedder'):
        """
        Initialize the embedder wrapper.
        
        Args:
            batch_size: Maximum number of sentences to process in a single batch
            embedder_path: Path to the embedder executable
        """
        # Set up environment once
        self.env = os.environ.copy()
        self.env['EMBEDDING_MODEL_PATH'] = os.path.expanduser('~/.cache/huggingface/hub/models--Xenova--nomic-embed-text-v1/snapshots/0b85f78966a655763985a595b770f221374dda10')
        self.default_batch_size = batch_size
        self.embedder_path = embedder_path
        
    def _process_single_sentence(self, sentence):
        """Process a single sentence through the embedder"""
        try:
            # Call the command line tool and get output
            result = subprocess.run(
                [self.embedder_path, sentence],
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
    
    def _process_batch_sentences(self, sentences):
        """Process multiple sentences using batch mode with null delimiter"""
        try:
            # Create null-delimited input for batch processing
            batch_input = '\0'.join(sentences) + '\0'
            
            # Call the embedder in batch mode
            result = subprocess.run(
                [self.embedder_path, '--batch'],
                input=batch_input,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                env=self.env,
                check=True
            )
            
            # Parse the output - each line is one embedding
            lines = result.stdout.strip().split('\n')
            embeddings = []
            
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    emb = [float(x) for x in line.split()]
                    
                    # Validate embedding quality
                    if len(emb) == 0:
                        raise ValueError(f"Empty embedding returned for sentence {i}")
                    if any(np.isnan(emb)) or any(np.isinf(emb)):
                        raise ValueError(f"Embedding contains NaN or Inf values for sentence {i}")
                    
                    embeddings.append(emb)
                except ValueError as e:
                    print(f"Error parsing embedding for sentence {i}: {e}")
                    print(f"Raw line: {repr(line)}")
                    raise
            
            # Verify we got the expected number of embeddings
            if len(embeddings) != len(sentences):
                raise ValueError(f"Expected {len(sentences)} embeddings, got {len(embeddings)}")
            
            return embeddings
            
        except subprocess.CalledProcessError as e:
            print(f"Error running batch embedder: {e}")
            print(f"stderr: {e.stderr}")
            print(f"Input sentences count: {len(sentences)}")
            raise
        except Exception as e:
            print(f"Error processing batch embeddings: {e}")
            raise
    
    def encode(self, sentences, batch_size=None, **kwargs):
        """
        Encode sentences using efficient batch processing.
        
        Args:
            sentences: Single string or list of strings to encode
            batch_size: Override default batch size for this call
            **kwargs: Additional arguments (for compatibility)
        
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Use provided batch_size or fall back to default
        effective_batch_size = batch_size or self.default_batch_size
        
        # For single sentences, use single processing for minimal overhead
        if len(sentences) == 1:
            return [self._process_single_sentence(sentences[0])]
        
        # For multiple sentences, prefer batch processing
        if len(sentences) <= effective_batch_size:
            # Process all at once if within batch size
            return self._process_batch_sentences(sentences)
        else:
            # Process in chunks if larger than batch_size
            all_embeddings = []
            for i in range(0, len(sentences), effective_batch_size):
                batch = sentences[i:i + effective_batch_size]
                batch_embeddings = self._process_batch_sentences(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings