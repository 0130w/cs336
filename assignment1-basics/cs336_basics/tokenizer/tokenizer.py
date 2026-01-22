from typing import Iterable, Iterator
from cs336_basics.tokenizer.utils import pretokenize

class Tokenizer:
  def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
    self.vocab = vocab
    self.merges = merges
    self.special_tokens = special_tokens
    return

  def __merge_word(self, word: str, merges: list[tuple[bytes, bytes]]) -> list[int]:
    word_token_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
    pass

  @classmethod
  def from_files(cls, vocab_filepath : str, merges_filepath : str, special_tokens : list[str] | None =None):
    return
  
  def encode(self, text: str) -> list[int]:
    word_list = pretokenize(text, self.special_tokens)
    encoding_list : list[int] = []
    for word in word_list:
      encoding_list += self.__merge_word(word, self.merges)
    return encoding_list
  
  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    pass

  def decode(self, ids: list[int]) -> str:
    pass