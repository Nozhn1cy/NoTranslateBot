import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.vocab import vocab

import random
import math
import time
import numpy as np

import matplotlib

matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import clear_output

from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter

from modules import Encoder, Decoder, Seq2Seq

# Determination
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# Tokenizer
tokenizer_W = WordPunctTokenizer()
def tokenize_ru(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())

def tokenize_en(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())

# Dataset 1
# dataset = []
# with open('data.txt', 'r', encoding="utf-8") as f:
#     for index, line in enumerate(f):
#         sents = line.rstrip().lower().split('\t')
#         dataset.append({'en': sents[0], 'ru' : sents[1]})

# Dataset2
dataset2 = []
with open('corpus.en_ru.1m.en', 'r', encoding="utf-8") as f:
    for line in f:
        sent = line.rstrip().lower()
        dataset2.append(sent)

dataset3 = []
with open('corpus.en_ru.1m.ru', 'r', encoding="utf-8") as f:
    for line in f:
        sent = line.rstrip().lower()
        dataset3.append(sent)
dataset = [{'en': sent_en, 'ru': sent_ru} for sent_en, sent_ru in zip(dataset2[0:200000], dataset3[0:200000])]
new_dataset = []
for pair in dataset:
    if len(tokenize_en(pair['en'])) < 30:
        new_dataset.append(pair)
print('Length of the longest en sentence: ',
      len(tokenize_en(sorted(new_dataset, key=lambda sentence: len(tokenize_en(sentence['en'])))[-1]['en'])))
print('Length of the longest ru sentence: ',
      len(tokenize_ru(sorted(new_dataset, key=lambda sentence: len(tokenize_ru(sentence['ru'])))[-1]['ru'])))
print(f'Dataset consists {format(len(new_dataset))} sentenses')
dataset = new_dataset
print('Length of the longest en sentence: ',
      len(tokenize_en(sorted(dataset, key=lambda sentence: len(tokenize_en(sentence['en'])))[-1]['en'])))
print('Length of the longest ru sentence: ',
      len(tokenize_ru(sorted(dataset, key=lambda sentence: len(tokenize_ru(sentence['ru'])))[-1]['ru'])))
print(f'Dataset consists {format(len(dataset))} sentenses')
data_train, data_val_test = train_test_split(dataset, test_size=0.2, random_state=SEED)
data_val, data_test = train_test_split(data_val_test, test_size=0.5, random_state=SEED)

# Vocabulary
counter1, counter2 = Counter(), Counter()
for line in data_train:
    counter1.update(tokenize_en(line['en']))
    counter2.update(tokenize_ru(line['ru']))

vocab_en = vocab(counter1, min_freq=3, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
vocab_ru = vocab(counter2, min_freq=3, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
vocab_en.set_default_index(vocab_en['<unk>'])
vocab_ru.set_default_index(vocab_ru['<unk>'])
print('En vocab constists {} tokens'.format(len(vocab_en)))
print('Ru vocab constists {} tokens'.format(len(vocab_ru)))
torch.save(vocab_en, 'vocab_en_ya.pt')  # define saved name
torch.save(vocab_ru, 'vocab_ru_ya.pt')  # define saved name
text_transform_en = lambda x: [vocab_en['<BOS>']] + [vocab_en[token] for token in tokenize_en(x)] + [vocab_en['<EOS>']]
text_transform_ru = lambda x: [vocab_ru['<BOS>']] + [vocab_ru[token] for token in tokenize_ru(x)] + [vocab_ru['<EOS>']]

# Dataloaders

batch_size = 32

def collate_batch(batch):
    en_list, ru_list = [], []
    for i, b in enumerate(batch):
        processed_text_en = torch.tensor(text_transform_en(b['en']))
        en_list.append(processed_text_en)
        processed_text_ru = torch.tensor(text_transform_ru(b['ru']))
        ru_list.append(processed_text_ru)
    return pad_sequence(en_list, padding_value=3.0).permute(1, 0), \
           pad_sequence(ru_list, padding_value=3.0).permute(1, 0)

def batch_sampler(data):
    indices = [(i, len(sent['en'])) for i, sent in enumerate(data)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
    pooled_indices = [x[0] for x in pooled_indices]
    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]

train_dataloader = DataLoader(data_train, batch_sampler=batch_sampler(data_train),
                              collate_fn=collate_batch)
valid_dataloader = DataLoader(data_val, batch_sampler=batch_sampler(data_val),
                              collate_fn=collate_batch)
test_dataloader = DataLoader(data_test, batch_sampler=batch_sampler(data_test),
                             collate_fn=collate_batch)

# Training
INPUT_DIM = len(vocab_en)
OUTPUT_DIM = len(vocab_ru)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
MAX_LENGTH = 62 #+2!

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device,
              max_length=MAX_LENGTH)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device,
              max_length=MAX_LENGTH)

SRC_PAD_IDX = vocab_en['<PAD>']
TRG_PAD_IDX = vocab_ru['<PAD>']

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights)

LEARNING_RATE = 5e-4
PATIENCE = 3
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# Train-evaluate
def train(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())
        # if (i + 1) % 10 == 0:
        #     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
        #
        #     clear_output(True)
        #     ax[0].plot(history, label='train loss')
        #     ax[0].set_xlabel('Batch')
        #     ax[0].set_title('Train loss')
        #     if train_history is not None:
        #         ax[1].plot(train_history, label='general train history')
        #         ax[1].set_xlabel('Epoch')
        #     if valid_history is not None:
        #         ax[1].plot(valid_history, label='general valid history')
        #     plt.legend()
        #     plt.show()
        len_iterator = i
    return epoch_loss / len_iterator

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)
            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            len_iterator = i
    return epoch_loss / len_iterator

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Training
train_history = []
valid_history = []

N_EPOCHS = 20
CLIP = 0.3

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_dataloader = DataLoader(data_train, batch_sampler=batch_sampler(data_train),
                                  collate_fn=collate_batch)
    valid_dataloader = DataLoader(data_val, batch_sampler=batch_sampler(data_val),
                                  collate_fn=collate_batch)
    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP, train_history, valid_history)
    valid_loss = evaluate(model, valid_dataloader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    scheduler.step(valid_loss)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-val-model.pt')

    train_history.append(train_loss)
    valid_history.append(valid_loss)
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
print('finally')
test_dataloader = DataLoader(data_test, batch_sampler=batch_sampler(data_test),
                             collate_fn=collate_batch)

# model.load_state_dict(torch.load('best-val-model.pt'))
test_loss = evaluate(model, test_dataloader, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Inference

def translate_sentence(sentence, vocab_en, vocab_ru, model, device, max_len=110):
    model.eval()

    if isinstance(sentence, str):
        indexes = [vocab_en['<BOS>']] + [vocab_en[token] for token in tokenize_en(sentence)] + [vocab_en['<EOS>']]
    else:
        indexes = [vocab_en['<BOS>']] + [vocab_en[token] for token in sentence] + [vocab_en['<EOS>']]

    src = torch.LongTensor(indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src)

    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)

    trg_indexes = [vocab_ru['<BOS>']]

    for i in range(max_len):

        trg = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg)

        with torch.no_grad():
            output, attention = model.decoder(trg, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == vocab_ru['<EOS>']:
            break

    trg_tokens = [vocab_ru.lookup_token(i) for i in trg_indexes]
    return trg_tokens[1:], attention
