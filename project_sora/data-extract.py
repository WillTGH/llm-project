import os
import lzma
from tqdm import tqdm
import time
import argparse
import bpe_tokenizer
import regex as re

parser = argparse.ArgumentParser(description='data-extract')
parser.add_argument('-charlvl', type=str, required=True, help='1 for char level, 0 for sub word:')
arg = parser.parse_args()
print(f'Tokenizer Character level 1/0: {arg.charlvl}')

charlvl = int(arg.charlvl)

bpe = bpe_tokenizer.BPETokenizer(vocab_size=1000000)
bpe.tokenizer_train()

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "V:/llm-project/datasets/test"

prefix = 'sw_'
output_file_train = f'{prefix}train_split.txt'
output_file_val = f'{prefix}val_split.txt'
vocab_file = f'{prefix}vocab.txt'

files = xz_files_in_dir(folder_path)

total_files = len(files)

split_index = int(total_files * 0.9) # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

print(f'0 for character level, 1 for sub-word level: {charlvl}')

start_time = time.time()

with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()

            if(charlvl == 1):
                #character level
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)
            else:
                #sub-word level
                outfile.write(text)
                sub_word = bpe.tokenize(text)
                # regex = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
                # rgx = re.findall(regex, sub_word) #regex get words and break things like 12d124d1d4g3g
                sb = set(sub_word)
                vocab.update(sb)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"{elapsed_time:.8f}")

start_time = time.time()

with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path,filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()

            if(charlvl == 1):
                #character level
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)
            else:
                #sub-word level
                outfile.write(text)
                sub_word = bpe.tokenize(text)
                # regex = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
                # rgx = re.findall(regex, sub_word) #regex get words and break things like 12d124d1d4g3g
                sb = set(sub_word)
                vocab.update(sb)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"{elapsed_time:.8f}")

with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')