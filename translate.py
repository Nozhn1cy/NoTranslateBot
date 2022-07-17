import torch
from nltk.tokenize import WordPunctTokenizer
from modules import Encoder, Decoder, Seq2Seq

# Tokenizer
tokenizer_W = WordPunctTokenizer()

def tokenize_ru(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())

def tokenize_en(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())

# Vocabulary
vocab_en = torch.load('vocab_en_ya.pt')
vocab_ru = torch.load('vocab_ru_ya.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
MAX_LENGTH = 62

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device,
              MAX_LENGTH
              )

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device,
              MAX_LENGTH)

SRC_PAD_IDX = vocab_en['<PAD>']
TRG_PAD_IDX = vocab_ru['<PAD>']

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load('best-val-model.pt'))

def translate_sentence(sentence, max_len=MAX_LENGTH):
    model.eval()
    print(sentence)
    if isinstance(sentence, str):
        indexes = [vocab_en['<BOS>']] + [vocab_en[token] for token in tokenize_en(sentence)] + [vocab_en['<EOS>']]
    else:
        indexes = [vocab_en['<BOS>']] + [vocab_en[token] for token in sentence] + [vocab_en['<EOS>']]
    print(indexes)
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
    print(trg_tokens[1:])
    a = merge(trg_tokens[1:-1], sentence)
    return a, attention

def merge(tokens, sentence):
    string = str()
    for i, token in enumerate(tokens):
        if token==',' or token=='.' or token==':':
            string += token
        elif token=='<unk>':
            if i < len(tokenize_en(sentence)):
                print(i, tokenize_en(sentence)[i])
                string += ' ' + tokenize_en(sentence)[i]
                print(string)
            else:
                break
        else:
            string = string + ' ' + token
    return string


