import os
import logging
from typing import Tuple, BinaryIO
import regex as re
from collections import defaultdict

INIT_VOCAB_SIZE = 256
MINI_CHUNK_SIZE = 4096
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

def find_chunk_boundaries(
    file : BinaryIO,
    special_tokens_bytes : list[bytes],
    desired_num_of_chunks
) -> list[int]:
  file.seek(0, os.SEEK_END)
  file_size = file.tell()
  file.seek(0)

  chunk_size = file_size // desired_num_of_chunks

  guess_chunk_boundaries = [i * chunk_size for i in range(desired_num_of_chunks)]
  guess_chunk_boundaries[-1] = file_size
  chunk_boundaries = []

  for bi in range(1, len(guess_chunk_boundaries)):
    if guess_chunk_boundaries[bi - 1] == file_size:
      break
    chunk_boundaries.append(guess_chunk_boundaries[bi - 1])
    if guess_chunk_boundaries[bi] < guess_chunk_boundaries[bi - 1]:
      guess_chunk_boundaries[bi] = guess_chunk_boundaries[bi - 1] + 1
    init_pos = guess_chunk_boundaries[bi]
    file.seek(guess_chunk_boundaries[bi])
    while True:
      mini_chunk = file.read(MINI_CHUNK_SIZE)
      if not mini_chunk:
        guess_chunk_boundaries[bi] = file_size
        break
      found_at = -1
      for special_token_bytes in special_tokens_bytes:
        find_pos = mini_chunk.find(special_token_bytes)
        if find_pos != -1:
          found_at = find_pos if found_at == -1 else min(found_at, find_pos)
      if found_at != -1:
        guess_chunk_boundaries[bi] = init_pos + found_at
        break
      init_pos += MINI_CHUNK_SIZE

  if chunk_boundaries and chunk_boundaries[len(chunk_boundaries) - 1] < file_size:
    chunk_boundaries.append(file_size)

  return chunk_boundaries

def train_bpe(
  input_path: str | os.PathLike,
  vocab_size: int,
  special_tokens: list[str],
  **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  assert(vocab_size >= INIT_VOCAB_SIZE)
  vocab : dict[int, bytes] = {i : bytes([i]) for i in range(0, INIT_VOCAB_SIZE)}
  merges : list[tuple[bytes, bytes]] = []
  desired_num_of_chunks = 16

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
    chunk_boundaries = find_chunk_boundaries(f, special_tokens_bytes, desired_num_of_chunks)
    freq_table = defaultdict(int)
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
      f.seek(start)
      chunk = f.read(end - start).decode("utf-8", errors="ignore")
      chunk_list = re.split("|".join(re.escape(special_token) for special_token in special_tokens), chunk) if special_tokens else [chunk]
      for chunk_item in chunk_list:
        for word in re.finditer(PAT, chunk_item):
          key = tuple(bytes([b]) for b in word.group().encode("utf-8"))
          freq_table[key] += 1

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