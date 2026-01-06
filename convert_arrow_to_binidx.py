#!/usr/bin/env python3
"""
Convert Arrow format dataset (saved with datasets library) to binidx format.
Usage: python convert_arrow_to_binidx.py <arrow_dataset_path> <output_path> [--tokenizer TOKENIZER] [--column_name COLUMN] [--ctxlen CTXLEN]
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer
from src.binidx import MMapIndexedDataset

# MMapIndexedDatasetBuilder class (from make_data_hf.py)
class MMapIndexedDatasetBuilder(object):
    def __init__(self, bin_filename, dtype=np.int32):
        self._data_file = open(bin_filename, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
        self._token_count = 0
    
    def token_count(self):
        return self._token_count
    
    def add_docs(self, docs, doc_lens):
        assert docs[0].dtype == self._dtype
        batch = np.concatenate(docs)
        self._token_count += np.sum(doc_lens)
        self._data_file.write(batch.tobytes(order="C"))
        self._sizes += doc_lens
    
    def finalize(self, index_file):
        self._data_file.close()
        self._doc_idx = range(len(self._sizes))
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert Arrow dataset to binidx format")
    parser.add_argument('input_path', type=str, help='Path to Arrow dataset directory')
    parser.add_argument('output_path', type=str, help='Output path prefix (will create .bin and .idx files)')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen3-8B-Base', help='Tokenizer path')
    parser.add_argument('--column_name', type=str, default='text', help='Column name containing text')
    parser.add_argument('--ctxlen', type=int, default=2048, help='Context length for magic_prime calculation')
    parser.add_argument('--max_tokens', type=int, default=None, help='Maximum tokens to convert (default: all)')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.input_path}...")
    dataset = load_from_disk(args.input_path)
    
    # Handle splits
    if hasattr(dataset, 'keys') and args.split in dataset.keys():
        dataset = dataset[args.split]
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    token_dtype = np.int32 if tokenizer.vocab_size > 65536 else np.uint16
    print(f"Using dtype: {token_dtype}")
    
    # Check if dataset is already tokenized (has input_ids column)
    is_tokenized = 'input_ids' in dataset.column_names
    if is_tokenized:
        print("✓ Dataset is already tokenized (has 'input_ids' column)")
        # Use input_ids column and get tokenizer just for EOS token
        args.column_name = 'input_ids'
        use_tokenizer = False
    else:
        # Check column exists
        if args.column_name not in dataset.column_names:
            print(f"Error: Column '{args.column_name}' not found in dataset.")
            print(f"Available columns: {dataset.column_names}")
            return
        use_tokenizer = True
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build binidx
    print(f"Building binidx files: {args.output_path}.bin and {args.output_path}.idx")
    builder = MMapIndexedDatasetBuilder(f"{args.output_path}.bin", dtype=token_dtype)
    
    total_tokens = 0
    processed_examples = 0
    
    # Process in batches for efficiency
    batch_size = 1000
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing"):
        batch = dataset[i:i+batch_size]
        data = batch[args.column_name]
        
        encodeds = []
        lens = []
        
        for item in data:
            if item is None:
                continue
            
            if use_tokenizer:
                # Tokenize text
                if item == "":
                    continue
                encoded = tokenizer(item, add_special_tokens=False)['input_ids']
                encoded.append(tokenizer.eos_token_id)  # Add EOS token
            else:
                # Already tokenized - item is a list of token IDs
                if isinstance(item, list):
                    encoded = list(item)
                else:
                    # Handle numpy arrays or other formats
                    encoded = item.tolist() if hasattr(item, 'tolist') else list(item)
                
                # Add EOS token if not already present
                if len(encoded) == 0 or encoded[-1] != tokenizer.eos_token_id:
                    encoded.append(tokenizer.eos_token_id)
            
            encoded = np.asarray(encoded, dtype=token_dtype)
            encodeds.append(encoded)
            lens.append(len(encoded))
        
        if encodeds:
            builder.add_docs(encodeds, lens)
            total_tokens = builder.token_count()
            processed_examples += len(encodeds)
        
        # Check max_tokens limit
        if args.max_tokens and total_tokens >= args.max_tokens:
            print(f"Reached max_tokens limit: {args.max_tokens}")
            break
    
    # Finalize
    builder.finalize(f"{args.output_path}.idx")
    print(f"Conversion complete!")
    print(f"Processed {processed_examples} examples")
    print(f"Total tokens: {total_tokens}")
    
    # Verify and calculate magic_prime
    print("\n### Verifying result...")
    data = MMapIndexedDataset(args.output_path)
    data_len = len(data)
    data_size = len(data._bin_buffer) // data._index._dtype_size
    
    print(f"Dataset has {data_size} tokens, {data_len} documents")
    
    # Calculate magic_prime
    if data_size >= args.ctxlen * 3:
        n_chunk = int(data_size // args.ctxlen) - 1
        for i in range(n_chunk, 0, -1):
            if i % 3 == 2:
                if is_prime(i):
                    print(f"\n### magic_prime = {i} (for ctxlen {args.ctxlen})")
                    print(f'\n--my_exit_tokens {data_size} --magic_prime {i} --ctx_len {args.ctxlen}\n')
                    break
    else:
        print(f"Warning: Dataset too small for ctxlen {args.ctxlen} (need at least {args.ctxlen * 3} tokens)")

if __name__ == "__main__":
    main()

