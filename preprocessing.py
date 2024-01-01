import pandas as pd
import re 


with open("data/الأربعون النووية.txt", "r", encoding="utf-8") as f:
    Arabic_text = f.read()
print("Total number of character:", len(Arabic_text))

def clean_Arabic_text(text):
    # Remove all non-word characters (everything except numbers and letters)
    text = re.sub(r"[^\w\s]", '', text)
    # Remove non-Arabic characters
    text = re.sub(r"[^\u0621-\u064A\s]", '', text)
    # Remove all numbers
    text = re.sub(r"\d+", '', text)
    # Normalize Arabic diacritics
    text = re.sub(r"[\u064B-\u065F]", '', text)
    return text

Arabic_text = clean_Arabic_text(Arabic_text)

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', Arabic_text)

preprocessed = [item.strip() for item in preprocessed if item.strip()]

print("Total number of tokens:", len(preprocessed))


all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab #A
        self.int_to_str = {i:s for s,i in vocab.items()} #B
    def encode(self, text): #C
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids): #D
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #E
        return text
    

tokenizer = SimpleTokenizerV1(vocab)

text = '''
الحمد لله رب  والصلاة والسلام على محمد  وعلى آله 
'''

ids = tokenizer.encode(text)
print(ids)
words= tokenizer.decode(ids)
print(words)

all_words.extend(["<|endoftext|>", "<|UNK|>"])

vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int   #A
        else "<|UNK|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #B
        return text
    
text1 = "الحمد لله رب العالمين والصلاة والسلام على محمد  وعلى آله"
text2 = "أنا اسمي عمر"
text = " <|endoftext|> ".join([text1, text2])

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.decode(tokenizer.encode(text)))