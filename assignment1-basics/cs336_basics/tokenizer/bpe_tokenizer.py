import os
from typing import Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.tokenizer.utils import find_chunk_boundaries, process_single_chunk

INIT_VOCAB_SIZE = 256

def train_bpe(
  input_path: str | os.PathLike,
  vocab_size: int,
  special_tokens: list[str],
  **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  assert(vocab_size >= INIT_VOCAB_SIZE)
  vocab : dict[int, bytes] = {i : bytes([i]) for i in range(0, INIT_VOCAB_SIZE)}
  merges : list[tuple[bytes, bytes]] = []
  freq_table = defaultdict(int)

  # --- Ensure long term matches first ---
  special_tokens.sort(key=len, reverse=True)
  special_tokens_bytes : list[bytes] = []
  # Add special tokens
  for special_token in special_tokens:
    if len(vocab) == vocab_size:
      return vocab, merges
    special_token_bytes = special_token.encode("utf-8")
    vocab[len(vocab)] = special_token_bytes
    special_tokens_bytes.append(special_token_bytes)

  with open(input_path, 'rb') as f:
    chunk_boundaries = find_chunk_boundaries(f, special_tokens_bytes)

  tasks = []
  for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
    tasks.append((input_path, start, end, special_tokens))

  with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_single_chunk, *task) for task in tasks]
    for future in futures:
      result = future.result()
      for word_str, count in result.items():
        key = tuple(bytes([b]) for b in word_str.encode("utf-8"))
        freq_table[key] += count

  pair_freq_table : defaultdict[Tuple[bytes, bytes], int] = defaultdict(int)
  # --- Merge stage ---
  while len(vocab) < vocab_size:
    pair_freq_table.clear()
    for key, value in freq_table.items():
      for pair in zip(key, key[1:]):
        pair_freq_table[pair] += value

    if not pair_freq_table:
      return vocab, merges
    
    target_pair = max(pair_freq_table, key=lambda pair : (pair_freq_table[pair], pair))
    merges.append(target_pair)
    new_token = target_pair[0] + target_pair[1]
    vocab[len(vocab)] = new_token
    new_freq_table = defaultdict(int)
    token0, token1 = target_pair

    for word, count in freq_table.items():
      if token0 not in word or len(word) < 2:
        new_freq_table[word] += count
        continue
      new_word = []
      idx = 0
      while idx < len(word):
        if idx < len(word) - 1 and token0 == word[idx] and token1 == word[idx + 1]:
          new_word.append(word[idx] + word[idx + 1])
          idx += 2
        else:
          new_word.append(word[idx])
          idx += 1
      new_freq_table[tuple(new_word)] += count
    freq_table = new_freq_table
  return vocab, merges