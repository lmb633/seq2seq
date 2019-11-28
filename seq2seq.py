import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
from seq2seq.model import Encoder, Decoder, Seq2Seq
import spacy

import random
import math
import time
import os

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 1000
CLIP = 1
PRINT_TREQ = 10
CHECKPOINT = 'BEST_checkpoint.tar'


def gen_data():
    global spacy_de, spacy_en, SRC, TRG, train_data, valid_data, test_data, device, train_iterator, valid_iterator, test_iterator
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings (tokens) and reverses it
        """
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC, TRG))
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    print(len(train_data.examples))
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
    BATCH_SIZE = 128
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train(model, iterator, optimizer, criterion, clip, epoch):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if i % PRINT_TREQ == 0:
            print('epoch:{0} ({1}/{2})  loss:{3} '.format(epoch, i, len(iterator), epoch_loss / (i + 1)))
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    file = open('result', 'wb')
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)  # turn off teacher forcing
            print(src.shape)
            print(trg.shape)
            print(output.shape)

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(epoch, epochs_since_improvement, model, best_loss, is_best, optimizer=None):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': best_loss,
             'model.py': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')
    return


def train_net():
    gen_data()
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    epochs_since_improvement = 0
    if os.path.exists(CHECKPOINT):
        print('load checkpoint')
        checkpoint = torch.load(CHECKPOINT)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model.py']
    else:
        print('train from beginning')
        start_epoch = 0
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
        model = Seq2Seq(enc, dec, device).to(device)

    # global epoch, train_loss, valid_loss, epoch_mins, epoch_secs, test_loss

    best_valid_loss = float('inf')
    model.apply(init_weights)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model.py has {count_parameters(model.py):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), weight_decay=0.005)

    PAD_IDX = TRG.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(start_epoch, N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, epoch)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_improvement = 0
            save_checkpoint(epoch, epochs_since_improvement, model, best_valid_loss, 1)

        else:
            epochs_since_improvement += 1
            print('Epochs since last improvement: ', epochs_since_improvement)
            save_checkpoint(epoch, epochs_since_improvement, model, best_valid_loss, 0)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    train_net()
    # gen_data()
