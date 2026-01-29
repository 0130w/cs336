import heapq
from typing import Iterable, Iterator
from cs336_basics.tokenizer.utils import pretokenize

class Tokenizer:
  def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
    self.vocab = vocab
    self.merges = merges
    self.special_tokens = special_tokens
    self.merges_dict = { merges[i] : i  for i in range(len(merges)) }
    self.reverse_vocab = { token_bytes : token_id for token_id, token_bytes in vocab.items() }
    return

  def __merge_word(self, word: str) -> list[int]:
    word_token_list = [bytes([b]) for b in word.encode("utf-8")]
    if len(word_token_list) <= 1:
      return [self.reverse_vocab[token_bytes] for token_bytes in word_token_list]
    prev = { i : i - 1 for i in range(len(word_token_list)) }
    next = { i : i + 1 for i in range(len(word_token_list)) }
    next[len(word_token_list) - 1] = -1 
    queue = []
    def __add_pair(i, j):
      pair = (word_token_list[i], word_token_list[j])
      if pair in self.merges:
        heapq.heappush(queue, (self.merges_dict[pair], i, j, word_token_list[j]))
    for i in range(len(word_token_list) - 1):
      __add_pair(i, i + 1)
    while queue:
      _, i, j, val = heapq.heappop(queue)
      if prev[i] == next[i] == -1 or word_token_list[j] != val:
        continue
      word_token_list[i] += word_token_list[j]
      next[i] = next[j]
      prev[j] = next[j] = -1 # set word_token_list[j] invalid
      if next[i] != -1:
        __add_pair(i, next[i])
      if prev[i] != -1:
        __add_pair(prev[i], i)
    result_ids = []
    curr = 0
    while curr != -1:
      result_ids.append(self.reverse_vocab[word_token_list[curr]])
      curr = next[curr]
    return result_ids

  @classmethod
  def from_files(cls, vocab_filepath : str, merges_filepath : str, special_tokens : list[str] | None = None):
    return

  # TODO: use cache to store ids
  def encode(self, text: str) -> list[int]:
    word_list = pretokenize(text, self.special_tokens)
    encoding_list : list[int] = []
    for word in word_list:
      ids = self.__merge_word(word)
      encoding_list += ids
    return encoding_list
  
  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    for text in iterable:
      yield from self.encode(text)

  def decode(self, ids: list[int]) -> str:
    return b"".join(self.vocab.get(id, b"") for id in ids).decode("utf-8")