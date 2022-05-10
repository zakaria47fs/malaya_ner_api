'''
!pip install malaya
!pip install youtokentome
!pip install -U sacremoses
'''

import malaya
import sacremoses
from tensorflow import errors

USE_GPU = False

device = 'CPU:0'
if USE_GPU and malaya.utils.available_gpu():
    device = 'GPU:0'

tokenizer = malaya.tokenizer.Tokenizer()
s_tokenizer = malaya.tokenizer.SentenceTokenizer()
detok = sacremoses.MosesDetokenizer('en')

    
ms_model = malaya.entity.transformer(model='albert', device=device)


def text_entities(text):

    text_entities = []

    try:
        sent_entities = ms_model.analyze(text)

        for ent in sent_entities:
            if ent['type'] in ['location', 'person', 'organization']:
                text_entities.append({'text': detok.detokenize(ent['text']), 'label': ent['type']})

        return text_entities
  
    except errors.InvalidArgumentError as e:
    
        sentences = [text]
        limit_tokens = 400

        if len(tokenizer.tokenize(text))>limit_tokens:
            sentences = s_tokenizer.tokenize(text)
            temp_sentences_list = []

            for sent in sentences:
                sent_tokens = tokenizer.tokenize(sent)
                
                # In case sub sentence has more than limit_tokens tokens
                if len(sent_tokens)>limit_tokens:
                    chunks = [sent_tokens[x:x+limit_tokens] for x in range(0, len(sent_tokens), limit_tokens)]
                    temp_sentences_list.extend([detok.detokenize(sent_tokens) for sent_tokens in chunks])

                else:
                    temp_sentences_list.append(sent)

            sentences = temp_sentences_list   

        for sent in sentences:
            sent_entities = ms_model.analyze(sent)

            temp_ents = []
            for ent in sent_entities:
                if ent['type'] in ['location', 'person', 'organization']:
                    temp_ents.append({'text': detok.detokenize(ent['text']), 'label': ent['type']})

            text_entities.extend(temp_ents)

    return text_entities


if __name__=='__main__':
    string = 'Menurut kisah lama orang Melayu, ketika Hang Jebat mengamuk dan segala usaha menghalangnya gagal'
    output = text_entities(string)
    print(output)
