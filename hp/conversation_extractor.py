import argparse
import langchain
from langchain import OpenAI
from typing import List
from .ingest import get_corpus
from chains.ExtractDialogChain import ExtractDialogChain, default_pre_prompt_dialogs

langchain.verbose = True

def get_conversations(non_message_padding: int) -> List[List[str]]:
    lines = get_corpus()

    # lines that start with " are dialog. we want to extract all dialog
    # lines and add args.conversation_padding lines of context before and after.
    # we group the dialog lines into conversations if there is no args.conversation_padding
    # lines of context between them.
    conversations = []
    conversation_lines = []
    conversation_start_index = None
    conversation_last_index = None

    index = 0
    for index, line in enumerate(lines):
        if line.startswith('"'):
            conversation_last_index = index
            if conversation_start_index is None:
                conversation_start_index = index - non_message_padding
                before_context = lines[conversation_start_index:conversation_start_index + non_message_padding]

                conversation_lines = before_context 

        if conversation_start_index and index - conversation_last_index > non_message_padding:
            conversation_start_index = None
            conversations.append(conversation_lines)
            conversation_lines = []

        conversation_lines.append(line)
    
    if conversation_start_index and conversation_lines:
        conversations.append(conversation_lines)
    
    return conversations

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--non-message-padding', type=int, default=2)
    args = parser.parse_args()

    conversations = get_conversations(args.non_message_padding)
    chain = ExtractDialogChain.from_llm(
        llm=OpenAI(
            temperature=0,
            # model_name='text-curie-001',
            max_tokens=500,
            
        ),
        # verbose=True,
    )

    # for conversation in conversations:
    #     print('\n\n\n\n')
    #     for line in conversation:
    #         print(line)
    #         print('')

    # print('\n\n\n\n')
    # print('\n\n\n\n')
    # print('\n\n\n\n')
    # print('\n\n\n\n')
    # print('\n\n\n\n')
    # print(chain.pre_prompt_dialogs)
    
    conversation = conversations[0]
    conversation = '\n\n'.join(conversation)
    print(conversation)
    
    messages = chain({
        'dialog': conversation,
    })['messages']

    for message in messages:
        print(message['speaker'], ':', message['message'], '\n')



if __name__ == '__main__':
    run()
