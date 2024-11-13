import os
import argparse
from datetime import datetime
import secrets
import copy
import tqdm
from tqdm import auto
from GPT2RGA import *
from miditok import REMI, CPWord
from miditoolkit import MidiFile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel


class EPianoDataset(Dataset):
    def __init__(self, token_data, max_seq_len, random_seq=True):
        self.token_data = token_data
        self.max_seq_len = max_seq_len
        self.random_seq = random_seq

        if len(self.token_data) < self.max_seq_len:
            raise ValueError("Token data length must be at least as long as max_seq_len.")

    def __len__(self):
        if self.random_seq:
            return len(self.token_data) // self.max_seq_len
        else:
            return (len(self.token_data) - 1) // self.max_seq_len

    def __getitem__(self, idx):
        if self.random_seq:
            start_idx = torch.randint(0, len(self.token_data) - self.max_seq_len - 1, (1,)).item()
        else:
            start_idx = idx * self.max_seq_len

        x = self.token_data[start_idx:start_idx + self.max_seq_len]
        y = self.token_data[start_idx:start_idx + self.max_seq_len]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LrStepTracker:
    def __init__(self, d_model, warmup_steps, init_step=0):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = init_step

    def step(self, step=None):
        if step is None:
            self.step_num += 1
            step = self.step_num
        else:
            self.step_num = step

        step = max(step, 1)
        lr = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return lr


def data_preprocessing(batch_size, use_chord):
    print('Tokenizing all MIDI files in the dataset...')
    dataset_path = './Pop1K7/midi_analyzed'

    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {
        'Chord': use_chord,
        'Rest': True,
        'Tempo': True,
        'Program': False,
        'TimeSignature': True,
        'rest_range': (2, 8),
        'nb_tempos': 32,
        'tempo_range': (40, 250)
    }

    tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)

    train_data = []
    midi_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.mid')])

    for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
        midi_path = os.path.join(dataset_path, midi_file)
        midi = MidiFile(midi_path)
        tokens = tokenizer.midi_to_tokens(midi)
        train_data.extend(tokens[0])

    print('Done!')

    val_data = train_data[:int(len(train_data) * 0.1)]
    train_data = train_data[int(len(train_data) * 0.1):]

    n_workers = 4
    max_seq = 1024

    train_dataset = EPianoDataset(train_data, max_seq)
    val_dataset = EPianoDataset(val_data, max_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers)

    print('Total INTs in the train dataset:', len(train_data))
    print('Total unique INTs in the dataset:', len(set(train_data)))
    print('Max INT in the dataset:', max(train_data))
    print('Min INT in the dataset:', min(train_data))
    print('=' * 50)

    return train_loader, val_loader, len(tokenizer.vocab.token_to_event)


def train(args):
    print('MidiTok Model Trainer')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.epochs

    train_loader, val_loader, vocab_size = data_preprocessing(args.batch_size, args.use_chord)

    config = GPT2Config(vocab_size=vocab_size, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12)
    model = GPT2LMHeadModel(config).to(device)

    d_model = config.n_embd
    SCHEDULER_WARMUP_STEPS = 4000
    LR_DEFAULT_START = 2e-5
    ADAM_BETA_1 = 0.9
    ADAM_BETA_2 = 0.95
    ADAM_EPSILON = 1e-8
    TOKEN_PAD = 0

    init_step = 0
    lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
    train_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    opt = Adam(model.parameters(), lr=LR_DEFAULT_START, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    lr_scheduler = LambdaLR(opt, lr_stepper.step)

    os.makedirs(args.checkpoint_folder, exist_ok=True)
    best_eval_loss = float("inf")
    loss_train, loss_val = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            opt.step()
            lr_scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        loss_train.append(avg_train_loss)

        # Evaluation
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(val_loader)
        loss_val.append(avg_eval_loss)

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_folder, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        # Save the best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_checkpoint_path = os.path.join(args.checkpoint_folder, "best_model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_eval_loss:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_train, 'b', label="Training Loss")
    plt.plot(loss_val, 'r', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_folder, 'Training_Validation_Loss.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Folder to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_chord", action='store_true', help="Enable chord tokenization in the dataset")
    args = parser.parse_args()

    train(args)
