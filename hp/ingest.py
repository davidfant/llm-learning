import os
import re
import requests
from typing import List
import faiss
import pickle
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

txt_path = './hp/data/book.txt'
txt_url = "https://raw.githubusercontent.com/amephraim/nlp/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%201%20-%20Sorcerer's%20Stone.txt"

def get_corpus() -> List[str]:
    if not os.path.exists(txt_path):
        with open(txt_path, 'w') as f:
            f.write(requests.get(txt_url).text)
    
    with open(txt_path, 'r') as f:
        return f.read().splitlines()

def process_corpus(lines: List[str]) -> str:
    # replace all caps lines with empty string
    lines = [re.sub(r'^[A-Z ]+$', '', line) for line in lines]
    # strip lines
    lines = [line.strip() for line in lines]

    # combine lines without empty lines between them
    paragraphs = []
    paragraph = ''
    for line in lines:
        if line:
            paragraph += line + ' '
        elif paragraph:
            paragraphs.append(paragraph)
            paragraph = ''

    return paragraphs


def create_chunks(lines: List[str]) -> List[str]:
    text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
    # split text into chunks and add them to an array
    chunks = []
    for line in lines:
        line_chunks = text_splitter.split_text(line)
        print(len(line_chunks), len(line))
        chunks.extend(line_chunks)
    
    return chunks


def run():
    lines = get_corpus()
    # lines = lines[:lines.index('CHAPTER TWO')]
    lines = process_corpus(lines)
    lines = ['\n'.join(lines)]
    chunks = create_chunks(lines)

    metadatas = [{ 'source': f'p-{i}' } for i in range(len(chunks))]

    store = FAISS.from_texts(chunks, OpenAIEmbeddings(), metadatas=metadatas)
    faiss.write_index(store.index, "hp/data/faiss.index")
    store.index = None
    with open("hp/data/faiss.pkl", "wb") as f:
        pickle.dump(store, f)


if __name__ == '__main__':
    run()
