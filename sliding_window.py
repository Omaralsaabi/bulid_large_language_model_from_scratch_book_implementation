import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')

with open("data/الأربعون النووية.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[200:]

context_size = 10 #A
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

## OUTPUT
    """

GPT2 tokenizer 

x: [96, 26897, 148, 114, 42092, 25405, 148, 118, 12919, 26897]
y:      [26897, 148, 114, 42092, 25405, 148, 118, 12919, 26897, 39848]
[96] ----> 26897
[96, 26897] ----> 148
[96, 26897, 148] ----> 114
[96, 26897, 148, 114] ----> 42092
[96, 26897, 148, 114, 42092] ----> 25405
[96, 26897, 148, 114, 42092, 25405] ----> 148
[96, 26897, 148, 114, 42092, 25405, 148] ----> 118
[96, 26897, 148, 114, 42092, 25405, 148, 118] ----> 12919
[96, 26897, 148, 114, 42092, 25405, 148, 118, 12919] ----> 26897
� ----> ر
�ر ----> �
�ر� ----> �
�رض ---->  و
�رض و ----> م
�رض وم ----> �
�رض وم� ----> �
�رض ومغ ----> ا
�رض ومغا ----> ر
�رض ومغار ----> ب

Please note that the output is only characters because the GPT2 tokenizer is a character-level tokenizer for Arabic.

cl100k_base tokenizer

x: [14628, 10386, 26957, 57894, 16552, 56434, 16552, 17607, 70782, 30925]
y:      [10386, 26957, 57894, 16552, 56434, 16552, 17607, 70782, 30925, 66498]
[14628] ----> 10386
[14628, 10386] ----> 26957
[14628, 10386, 26957] ----> 57894
[14628, 10386, 26957, 57894] ----> 16552
[14628, 10386, 26957, 57894, 16552] ----> 56434
[14628, 10386, 26957, 57894, 16552, 56434] ----> 16552
[14628, 10386, 26957, 57894, 16552, 56434, 16552] ----> 17607
[14628, 10386, 26957, 57894, 16552, 56434, 16552, 17607] ----> 70782
[14628, 10386, 26957, 57894, 16552, 56434, 16552, 17607, 70782] ----> 30925
ت ----> م
تم ----> ة
تمة ---->  ل
تمة ل ----> ه
تمة له ----> ذ
تمة لهذ ----> ه
تمة لهذه ---->  ال
تمة لهذه ال ----> أ
تمة لهذه الأ ----> ح
تمة لهذه الأح ----> اد
    """