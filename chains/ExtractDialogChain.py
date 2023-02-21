from langchain.chains.base import Chain
from langchain import PromptTemplate, OpenAI
from langchain.llms.base import BaseLLM
from langchain.chains.llm import LLMChain
from typing import Dict, Any, List

message_prompt_template = """
=========
MESSAGE: {message}
SPEAKER: {speaker}
""".strip()

dialog_prompt_template = """
Your goal is to transcribe and reformat text and extract the dialog and speakers. Dialog always start and end with citation marks and you should only include that. Include both short, medium and long length dialog. Include all speakers in the dialog. If you do not know the speaker, include the message and set the speaker to "UNKNOWN". Do a very precise job, keep dialogs separate and include every message.

DIALOG:
{dialog}
MESSAGES:
{messages}

#########
""".strip()

extract_dialog_prompt_template = """
{pre_prompt_dialogs}

DIALOG:
{dialog}

MESSAGES:
========
""".strip()

message_prompt = PromptTemplate(template=message_prompt_template, input_variables=['speaker', 'message'])
dialog_prompt = PromptTemplate(template=dialog_prompt_template, input_variables=['dialog', 'messages'])
extract_dialog_prompt = PromptTemplate(template=extract_dialog_prompt_template, input_variables=['pre_prompt_dialogs', 'dialog'])

default_pre_prompt_dialogs = [{
  'dialog': """
WHAM -- a thud and a nasty crack and Neville lay facedown on the grass
in a heap.

Madam Hooch was bending over Neville, her face as white as his.

"Broken wrist," Harry heard her mutter. "Come on, boy -- it's all right,
up you get.".

She turned to the rest of the class and wondered. Do you understand what this means?

"None of you is to move while I take this boy to the hospital wing! Come on, dear."

Neville, his face tear-streaked, clutching his wrist, hobbled off with
Madam Hooch, who had her arm around him.

"That must have hurt!" Harry looked very sad. "Hope he gets better soon"

No sooner were they out of earshot than Malfoy burst into laughter.

"Did you see his face, the great lump?"

The other Slytherins joined in.

"Yea, that was crazy", said someone. "But he is not so smart so it is not surprising"

"What?" Ron seemed confused.

"Shut up, Malfoy," snapped Parvati Patil.

"Ooh, sticking up for Longbottom?" said Pansy Parkinson, a hard-faced
Slytherin girl. "Never thought you'd like fat little crybabies,
Parvati."
  """.strip(),
  'messages': [{
    'speaker': 'Madam Hooch',
    'message': 'Broken wrist. Come on, boy -- it\'s all right, up you get.',
  }, {
    'speaker': 'Madam Hooch',
    'message': 'None of you is to move while I take this boy to the hospital wing! Come on, dear.',
  }, {
    'speaker': 'Harry',
    'message': 'That must have hurt! Hope he gets better soon',
  }, {
    'speaker': 'Malfoy',
    'message': 'Did you see his face, the great lump?',
  }, {
    'speaker': 'UNKNOWN',
    'message': 'Yea, that was crazy. But he is not so smart so it is not surprising',
  }, {
    'speaker': 'Ron',
    'message': 'What?',
  }, {
    'speaker': 'Parvati Patil',
    'message': 'Shut up, Malfoy.',
  }, {
    'speaker': 'Pansy Parkinson',
    'message': 'Ooh, sticking up for Longbottom? Never thought you\'d like fat little crybabies, Parvati.',
  }]
}]
default_pre_prompt_dialogs = [{**d, 'messages': '\n'.join([message_prompt.format(**m) for m in d['messages']])} for d in default_pre_prompt_dialogs]
default_pre_prompt_dialogs = [dialog_prompt.format(**d) for d in default_pre_prompt_dialogs]
default_pre_prompt_dialogs = '\n\n'.join(default_pre_prompt_dialogs)


class ExtractDialogChain(Chain):

    extract_dialog_chain: LLMChain
    pre_prompt_dialogs: str = default_pre_prompt_dialogs

    @classmethod
    def from_llm(cls, llm: BaseLLM, **kwargs) -> 'ExtractDialogChain':
        extract_dialog_chain = LLMChain(llm=llm, prompt=extract_dialog_prompt)
        return cls(
            extract_dialog_chain=extract_dialog_chain,
            **kwargs,
        )
    
    @property
    def input_keys(self) -> List[str]:
        return ['dialog']

    @property
    def output_keys(self) -> List[str]:
        return ['messages']
    
    def _call(self, inputs: Dict[str, Any]) -> str:
        output = self.extract_dialog_chain({
            'pre_prompt_dialogs': self.pre_prompt_dialogs,
            'dialog': inputs['dialog'],
            'stop': '#########',
        })

        messages = []
        for line in output['text'].split('========='):
            line = line.strip()
            if not line:
                continue
            speaker = line.split('SPEAKER: ')[1].split('\n')[0]
            message = line.split('MESSAGE: ')[1].split('\n')[0]
            messages.append({ 'speaker': speaker, 'message': message })

        return { 'messages': messages }

class VerboseOpenAI(OpenAI):
    def _generate(self, *args, **kwargs):
        # for p in args[0]:
        #     print(p)
        return super()._generate(*args, **kwargs)

if __name__ == '__main__':
#     p = extract_dialog_prompt.format(
#         pre_prompt_dialogs=default_pre_prompt_dialogs,
#         dialog= """
# "Who is this?" said Harry, looking at the Remembrall.

# "Longbottom's gran sent it to him," said Malfoy, "to remind him to remember things."

# "Like what?" said Harry.
#    """.strip()
#      )

#     print('PPP', p)


    chain = ExtractDialogChain.from_llm(
        llm=VerboseOpenAI(
            temperature=0,
            # model_name='text-curie-001',
            max_tokens=500,
            
        ),
        verbose=True,
    )

    result = chain({
        'dialog': """
"Who is this?" said Harry, looking at the Remembrall.

"Longbottom's gran sent it to him," said Malfoy, "to remind him to remember things."

"Like what?" said Harry.
  """.strip(),
})

    print(result['messages'])

