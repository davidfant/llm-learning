import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse

parser = argparse.ArgumentParser(description='Ask a question about Harry Potter.')
parser.add_argument('question', type=str, help='The question to ask')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("hp/data/faiss.index")

with open("hp/data/faiss.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index


class VerboseOpenAI(OpenAI):
  def _generate(self, *args, **kwargs):
    for p in args[0]:
      print(p)
    print(kwargs)
    return super()._generate(*args, **kwargs)


chain = VectorDBQAWithSourcesChain.from_llm(
  # llm=OpenAI(temperature=0),
  llm=VerboseOpenAI(temperature=0),
  verbose=True,
  vectorstore=store,
)

inputs = {"question": args.question}
result = chain(inputs)

print(f"Answer: {result['answer']}")
