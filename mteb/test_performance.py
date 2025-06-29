#!/usr/bin/env python3

import time
import os
from my_sentence_transformer import MySentenceTransformer

# Set optimal thread count for Apple Silicon
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

def test_performance():
    model = MySentenceTransformer()
    
    # Test different batch sizes
    test_sentences = [
        "This is a test sentence for performance evaluation.",
        "Another sentence to test the embedding generation speed.",
        "We want to see how fast the CoreML optimized version runs.",
        "Apple Silicon should provide better performance than CPU.",
        "The Neural Engine can accelerate transformer computations.",
        "Let's benchmark the embedding generation speed.",
        "This test helps us understand the performance gains.",
        "Parallel processing can help with larger batches.",
    ]
    
    print("🚀 Testing Apple Silicon Optimized Embedder Performance")
    print("=" * 60)
    print(f"CPU Count: {os.cpu_count()}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    print()
    
    # Test single sentence
    print("📝 Single Sentence Test:")
    start_time = time.time()
    embeddings = model.encode(test_sentences[0])
    single_time = time.time() - start_time
    print(f"   Time: {single_time:.3f} seconds")
    print(f"   Embedding shape: {len(embeddings[0])} dimensions")
    print()
    
    # Test small batch
    print("📦 Small Batch Test (2 sentences):")
    start_time = time.time()
    embeddings = model.encode(test_sentences[:2])
    small_batch_time = time.time() - start_time
    print(f"   Time: {small_batch_time:.3f} seconds")
    print(f"   Throughput: {2/small_batch_time:.1f} sentences/second")
    print()
    
    # Test larger batch
    print("📦 Larger Batch Test (8 sentences):")
    start_time = time.time()
    embeddings = model.encode(test_sentences)
    large_batch_time = time.time() - start_time
    print(f"   Time: {large_batch_time:.3f} seconds")
    print(f"   Throughput: {8/large_batch_time:.1f} sentences/second")
    print()
    
    # Performance summary
    print("🎯 Performance Summary:")
    print(f"   Single sentence: {single_time:.3f}s")
    print(f"   Small batch (2): {small_batch_time:.3f}s ({2/small_batch_time:.1f} sent/s)")
    print(f"   Large batch (8): {large_batch_time:.3f}s ({8/large_batch_time:.1f} sent/s)")
    print()
    
    # Verify embedding quality
    print("✅ Embedding Quality Check:")
    first_embedding = embeddings[0]
    print(f"   Dimension: {len(first_embedding)}")
    print(f"   Range: [{min(first_embedding):.4f}, {max(first_embedding):.4f}]")
    print(f"   Mean: {sum(first_embedding)/len(first_embedding):.4f}")
    print(f"   Sample values: {first_embedding[:5]}")
    

if __name__ == "__main__":
    test_performance()
