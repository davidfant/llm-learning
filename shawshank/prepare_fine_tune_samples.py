import argparse
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--speaker', type=str, required=True)
  parser.add_argument('--context-length', type=int, default=5)
  parser.add_argument('--stop-sequence', type=str, default='\n###')
  args = parser.parse_args()

  with open('shawshank/data/conversations.json') as f:
    conversations = json.load(f)
  
  samples = []
  for conversation in conversations:
    lines = [l for l in conversation if l['mode'] is None]
    for index, line in enumerate(lines):
      if line['speaker'] == args.speaker:
        lines_before = lines[max(0, index - args.context_length):index]

        prompt = '\n\n'.join([f"{l['speaker']}: {l['content']}" for l in lines_before])
        prompt += f"\n\n{line['speaker']}:"
        completion = f" {line['content']}{args.stop_sequence}"

        print('\n\n\n\n')
        print(prompt)
        print(completion)
      
        samples.append({ 'prompt': prompt, 'completion': completion })

  output_path = f'shawshank/data/samples-{args.speaker}.jsonl'
  with open(output_path, 'w') as f:
    f.write('\n'.join([json.dumps(sample) for sample in samples]))
  
  # openai api fine_tunes.create -t shawshank/data/samples-ANDY.jsonl -m davinci --suffix "shawshank-andy"
