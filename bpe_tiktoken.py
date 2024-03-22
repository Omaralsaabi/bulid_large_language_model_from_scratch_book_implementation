import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')

text = '''
الحمد لله رب <|endoftext|> والصلاة والسلام على محمد  وعلى آله 
'''

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

unknown_word = "Akwirw ier"

token_ids = tokenizer.encode(unknown_word)
print("Token IDs:", token_ids)

for token_id in token_ids:
    print("Token ID:", token_id, "Token:", tokenizer.decode([token_id]))

unknown_word_decoded = tokenizer.decode(token_ids)
print("Unknown word decoded:", unknown_word_decoded)