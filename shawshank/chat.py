import faiss
import pickle

def run():
  index = faiss.read_index("shawshank/data/faiss.index")
  with open("shawshank/data/faiss.pkl", "rb") as f:
      store = pickle.load(f)
  store.index = index

  similar = store.similarity_search_with_score("How did your wife die?", 10)

  for doc, score in similar:
    print('Score: ', score)
    print(doc.page_content)
    print('\n\n')

  # using context
  

if __name__ == '__main__':
  run()
