import torch
import os

# General Configuration
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "aclImdb_v1.tar.gz")
EXTRACT_PATH = os.path.join(DATA_DIR, "aclImdb")

# Preprocessing Configuration
MAX_FEATURES_TFIDF = 5000

# Deep Learning Hyperparameters
MAX_VOCAB_SIZE = 10000
MAX_LENGTH = 250
BATCH_SIZE = 128
EMBED_DIM = 64
HIDDEN_DIM = 64
LEARNING_RATE = 0.01
EPOCHS = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')