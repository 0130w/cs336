import os
import regex as re
from typing import BinaryIO
from collections import defaultdict

MINI_CHUNK_SIZE = 65536 # 64KB
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file : BinaryIO,
    special_tokens_bytes : list[bytes]
) -> list[int]:
  file.seek(0, os.SEEK_END)
  file_size = file.tell()
  file.seek(0)
  desired_num_of_chunks = os.cpu_count()
  assert(desired_num_of_chunks is not None)

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

def pretokenize(
  content: str,
  special_tokens: list[str] | None
) -> list[str]:
  content_list = re.split("|".join(re.escape(special_token) for special_token in special_tokens), content) if special_tokens else [content]
  word_list = []
  for content_item in content_list:
    for word in re.finditer(PAT, content_item):
      word_str = word.group()
      word_list.append(word_str)
  return word_list

def get_freq_table(
    content: str,
    special_tokens: list[str] | None
) -> defaultdict[str, int]:
  """Run pretokenize process and return freq_table
  """
  local_freq_table = defaultdict(int)
  content_list = re.split("|".join(re.escape(special_token) for special_token in special_tokens), content) if special_tokens else [content]
  for content_item in content_list:
    for word in re.finditer(PAT, content_item):
      word_str = word.group()
      local_freq_table[word_str] += 1
  return local_freq_table

def process_single_chunk(
    input_path : str | os.PathLike,
    start : int,
    end : int,
    special_tokens : list[str]
) -> defaultdict[str ,int]:
  with open(input_path, "rb") as f:
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return get_freq_table(chunk, special_tokens)