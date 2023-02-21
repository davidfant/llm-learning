import os
import simplejson as json
from typing import List, NamedTuple, Optional, Tuple


class ConversationItem(NamedTuple):
  speaker: str
  content: str
  mode: Optional[str]
  start_index: int
  end_index: int
  continued: bool

def matches_indents(text: int, n: int) -> bool:
  # calculate number of leading tabs
  # indents = len(text) - len(text.lstrip('\t'))
  indents = len(text) - len(text.lstrip())
  return indents == n

def get_lines() -> List[str]:
  txt_path = './shawshank/data/book.txt'

  with open(txt_path, 'r') as f:
    lines = f.readlines() # [1033:1060]
    lines = [line.rstrip() for line in lines]

    # for l in lines:
    #   print({'l': l})
    #   print(matches_indents(l, 0), '\t', len(l), '\t', len(l.lstrip()), '\t', {'0': l, '1': l.lstrip()})
    #   print('\n')

    lines = [line for line in lines if line]
    lines = ['' if matches_indents(line, 0) else line for line in lines]
    lines = [line for line in lines if not (matches_indents(line, 3) and line.strip().startswith('('))]
    # lines = [line for line in lines if line]
    # combine lines that have the same indentation
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


def get_speaker_and_mode(line: str) -> Tuple[str, Optional[str], bool]:
  # convert strings like:
  # ANDY => ANDY, None, False
  # ANDY (CONT'D) => ANDY, None, True
  # ANDY (V.O.) => ANDY, 'V.O.', False

  line = line.strip()
  name = line.split(' ')[0]
  # continued = "CONT'D" in line
  continued = '(cont.)' in line
  if '(V.O.)' in line:
    mode = 'voiceover'
  elif '(O.S.)' in line:
    mode = 'offscreen'
  else:
    mode = None

  return name, mode, continued

def get_conversation_items(lines: List[str]) -> List[ConversationItem]:
  items: List[ConversationItem] = []

  index = 0
  while index < len(lines):
    # if line starts with 4 tabs
    if matches_indents(lines[index], 4):
      start_index = index
      speaker, mode, continued = get_speaker_and_mode(lines[index])

      # get lines with 2 indent until blank line
      # content = ''
      # index += 1
      # while index < len(lines) and lines[index].startswith(indent(2)):
      #   content += lines[index].strip() + ' '
      #   index += 1

      while index < len(lines):
        if matches_indents(lines[index], 2):
          content = lines[index].strip()
          # if not content.startswith('(') and not content.endswith(')'):
          item = ConversationItem(speaker, content, mode, start_index, index, continued)
          items.append(item)
          break

        index += 1
    index += 1
  
  return items
  

def merge_continued_conversations(items: List[ConversationItem]) -> List[ConversationItem]:
  # merge items that are continued
  merged = []
  for i in range(len(items)):
    if items[i].continued:
      merged[-1] = ConversationItem(
        merged[-1].speaker,
        merged[-1].content + ' ' + items[i].content,
        merged[-1].mode,
        merged[-1].start_index,
        items[i].end_index,
        False
      )
    else:
      merged.append(items[i])
  
  return merged


def group_conversations(items: List[ConversationItem], max_gap: int) -> List[List[ConversationItem]]:
  conversations: List[List[ConversationItem]] = []
  conversation: List[ConversationItem] = [items[0]]

  for i in range(1, len(items)):
    gap = items[i].start_index - conversation[-1].end_index
    if gap <= max_gap:
      # print(items[i], '\n')
      conversation.append(items[i])
    else:
      conversations.append(conversation)
      conversation = [items[i]]
  
  return conversations


def log_stats(conversations: List[List[str]]):
  convs_by_len = {}
  lines_by_speaker = {}

  for c in conversations:
    convs_by_len[len(c)] = convs_by_len.get(len(c), 0) + 1
    for i in c:
      lines_by_speaker[i.speaker] = lines_by_speaker.get(i.speaker, 0) + 1
  
  # sort keys
  convs_by_len = {k: v for k, v in sorted(convs_by_len.items(), key=lambda item: item[0])}
  lines_by_speaker = {k: v for k, v in sorted(lines_by_speaker.items(), key=lambda item: item[1], reverse=True)}
  
  print('Conversation Lengths:')
  for k, v in convs_by_len.items():
    print(k, v)
  print('\n')

  print('Lines by Speaker:')
  for k, v in lines_by_speaker.items():
    print(k, v)
  print('\n')


def run():
  lines = get_lines()

  with open('./shawshank/data/book_lines.txt', 'w') as f:
    f.write('\n\n'.join(lines))

  items = get_conversation_items(lines)
  items = merge_continued_conversations(items)
  conversations = group_conversations(items, 2)

  print('Lines:', len(lines))
  print('Items:', len(items))
  print('Items w/o mode:', len([i for i in items if not i.mode]))
  print('Conversations:', len(conversations))
  log_stats(conversations)

  with open('./shawshank/data/conversations.json', 'w') as f:
    json.dump(conversations, f, indent=2)

  # for i in items:
  #   print(i.start_index, i.end_index, i.speaker, i.mode)
  #   print(i.content)
  #   print('\n')


  # print(len(conversations))
  # for c in conversations:
  #   # if len(c) == 62:
  #     for l in c:
  #       print(l.start_index, l.end_index, l.speaker, l.mode, l.content)

  #     print('\n' * 4)

  # for l in lines:
  #   print(l, '\n')

  # for c in conversations:
  #   for i in c:
  #     print(i.speaker, i.mode)
  #     print(i.content)
  #     print('\n')
  #   print('\n' * 4)

  # print(len(items), len(conversations))

  # for c in items:
  #   print(c)
  #   print('\n')


if __name__ == '__main__':
  run()
