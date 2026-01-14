from typing import Iterable, Iterator

class Tokenizer:
  def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
    return

  def from_files(self):
    return
  
  def encode(self, text: str) -> list[int]:
    pass
  
  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    pass

  def decode(self, ids: list[int]) -> str:
    pass