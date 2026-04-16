import re
import torch
from torch.utils.data import Dataset
from collections import Counter
from src.config import MAX_VOCAB_SIZE, MAX_LENGTH

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text) # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)   # Remove punctuation
    return text

def tokenize(text):
    return clean_text(text).split()

def build_vocab(texts):
    all_words = []
    for text in texts:
        all_words.extend(tokenize(text))
    counts = Counter(all_words)
    vocab = {word: i+2 for i, (word, _) in enumerate(counts.most_common(MAX_VOCAB_SIZE-2))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def text_to_ints(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, 1) for token in tokens][:MAX_LENGTH]

def pad_sequence(seq):
    return seq + [0] * (MAX_LENGTH - len(seq))

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.data = [torch.tensor(pad_sequence(text_to_ints(t, vocab))) for t in texts]
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self): 
        return len(self.labels)
    
    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx]