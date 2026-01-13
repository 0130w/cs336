import os
import logging
from typing import Tuple
import regex as re
from collections import defaultdict

INIT_VOCAB_SIZE = 256

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

def train_bpe(
  input_path: str | os.PathLike,
  vocab_size: int,
  special_tokens: list[str],
  **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  assert(vocab_size >= INIT_VOCAB_SIZE)
  vocab : dict[int, bytes] = {i : bytes([i]) for i in range(0, INIT_VOCAB_SIZE)}
  merges : list[tuple[bytes, bytes]] = []

  # Add special tokens
  for special_token in special_tokens:
    if len(vocab) == vocab_size:
      return vocab, merges
    vocab[len(vocab)] = special_token.encode('utf-8')

  with open(input_path, 'r') as f:
    content = f.read()

  # --- Remove special tokens from text ---
  # --- Ensure long term matches first ---
  special_tokens.sort(key=len, reverse=True)
  content_list = re.split("|".join(re.escape(special_token) for special_token in special_tokens if special_token), content) if special_tokens else [content] 
  logging.debug(f'content_list = {content_list}, type = {type(content_list)}')

  PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

  freq_table = defaultdict(int)
  pair_freq_table : defaultdict[Tuple[bytes, bytes], int] = defaultdict(int)
  # --- Pre-tokenization ---
  for content_piece in content_list:
    for item in re.finditer(PAT, content_piece):
      key = tuple(bytes([b]) for b in item.group().encode("utf-8"))
      freq_table[key] += 1

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