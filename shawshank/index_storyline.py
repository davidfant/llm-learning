import re
import simplejson as json
import pickle
import faiss
from typing import List, NamedTuple, Optional, Tuple
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings


def matches_indents(text: int, n: int) -> bool:
  # calculate number of leading tabs
  # indents = len(text) - len(text.lstrip('\t'))
  indents = len(text) - len(text.lstrip())
  return indents == n


def get_scene_metadata(line: str) -> Tuple[str, str, str]:
  # remove leading and trailing number from strign
  line = re.sub(r'^\d+\s*', '', line)
  line = re.sub(r'\s*\d+$', '', line)
  return line.split(' -- ')


def get_scene_start(line: str) -> Optional[Tuple[str, str, str]]:
  if matches_indents(line, 0) and re.match(r'^\d+\s', line):
    metadata = get_scene_metadata(line)
    if len(metadata) == 3:
      return metadata
  if matches_indents(line, 1) and re.match(r'^{INT|EXT} --', line.strip()):
    metadata = get_scene_metadata(line)
    if len(metadata) == 2:
      return metadata
  return None


def merge_lines_with_same_indentation(lines: List[str]) -> List[str]:
  return_lines = []
  line = ''
  line_indentation = None
  for l in lines:
    current_line_indentation = len(l) - len(l.lstrip())

    if line_indentation != current_line_indentation:
      if line:
        return_lines.append(line)
      line = l
      line_indentation = current_line_indentation
    else:
      line += ' ' + l.strip()
  
  return return_lines


def merge_conversation_speaker_and_line(lines: List[str]) -> List[str]:
  return_lines = []

  i = 0
  while i < len(lines):
    line = lines[i]
    next_line = lines[i + 1] if i + 1 < len(lines) else None
    if matches_indents(line, 4) and next_line and matches_indents(next_line, 2):
      return_lines.append(line.strip() + ': ' + next_line.strip())
      i += 2
    else:
      return_lines.append(line.strip())
      i += 1
  
  return return_lines


def transform_scene_starts(lines: List[str]) -> List[str]:
  returned_lines = []
  for line in lines:
    scene_start = get_scene_start(line)
    if scene_start:
      intext, location, time = scene_start
      returned_lines.append('')
      returned_lines.append(f"Location: {location} | Time: {time}")
    else:
      returned_lines.append(line)
  return returned_lines


def get_lines() -> List[str]:
  txt_path = './shawshank/data/book.txt'

  with open(txt_path, 'r') as f:
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    lines = merge_lines_with_same_indentation(lines)
    lines = merge_conversation_speaker_and_line(lines)
    lines = [l for l in lines if l]
    lines = transform_scene_starts(lines)

    return lines


def run():
  lines = get_lines()
  text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")

  chunks = '\n'.join(lines).split('\n\n')
  chunks = [c for chunk in chunks for c in text_splitter.split_text(chunk)]

  metadatas = [{ 'source': f'ch-{i}' } for i in range(len(chunks))]

  store = FAISS.from_texts(chunks, OpenAIEmbeddings(), metadatas=metadatas)
  faiss.write_index(store.index, "shawshank/data/faiss.index")
  store.index = None
  with open("shawshank/data/faiss.pkl", "wb") as f:
    pickle.dump(store, f)


if __name__ == '__main__':
  run()
