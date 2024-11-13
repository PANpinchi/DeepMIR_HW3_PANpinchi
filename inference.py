import os
import argparse
from datetime import datetime
import secrets
import copy
import tqdm
from tqdm import auto
from GPT2RGA import *
from miditok import REMI, CPWord, get_midi_programs
from miditoolkit import MidiFile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel


def test(args):
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {
        'Chord': args.use_chord,
        'Rest': True,
        'Tempo': True,
        'Program': False,
        'TimeSignature': True,
        'rest_range': (2, 8),
        'nb_tempos': 32,
        'tempo_range': (40, 250)
    }

    # Creates the tokenizer and loads a MIDI
    tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)  # REMI encoding

    print('Loading the model...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = len(tokenizer.vocab.token_to_event)
    config = GPT2Config(vocab_size=vocab_size, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12)

    # Initialize GPT2LMHeadModel with configuration
    model = GPT2LMHeadModel(config).to(device)

    # Load model weights
    best_model_file = 'best_model.pth'  # 'best_model.pth' or 'epoch_150.pth'
    params_path = os.path.join(args.saved_dir, best_model_file)
    model.load_state_dict(torch.load(params_path))
    print('Done!')

    print('MidiTok Model Generator')

    model.eval()

    # Generate continuous MIDI files
    bar_token_id = tokenizer.vocab.event_to_token["Bar_None"]
    max_bars = 32
    number_of_prime_tokens = 128
    os.makedirs(args.output_dir, exist_ok=True)  # Ensure the output directory exists

    # Generate a sequence from the model
    midi = MidiFile(args.inputs)
    tokens1 = tokenizer.midi_to_tokens(midi)

    input_seed = torch.Tensor([tokens1[0]]).long().to(device)
    generated_sequence = model.generate(
        input_seed,
        max_length=1024,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        pad_token_id=0
    )

    # Convert tokens to MIDI
    out_tokens = generated_sequence[0].cpu().numpy().tolist()

    generated_bars = out_tokens.count(bar_token_id)

    out_all = [out_tokens]
    while generated_bars < max_bars:
        input_seed = torch.tensor([out_tokens[-number_of_prime_tokens:]]).to(device)
        generated_sequence = model.generate(
            input_seed,
            max_length=1024,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            pad_token_id=0
        )

        out_tokens_1 = generated_sequence[0].cpu().numpy().tolist()
        out_all.append(out_tokens_1[number_of_prime_tokens:])
        out_tokens = out_tokens_1[number_of_prime_tokens:]

        generated_bars = sum(seq.count(bar_token_id) for seq in out_all)

    flattened_out_all = [token for seq in out_all for token in seq]
    bar_indices = [i for i, token in enumerate(flattened_out_all) if token == bar_token_id]
    if len(bar_indices) > max_bars:
        flattened_out_all = flattened_out_all[:bar_indices[max_bars]]

    # Convert tokens back to MIDI
    midi_file = tokenizer.tokens_to_midi([flattened_out_all])
    file_name = os.path.basename(args.inputs)
    output_path = os.path.join(args.output_dir, file_name)

    midi_file.dump(output_path)

    print(f'Saved: {output_path}, generated_bars: {flattened_out_all.count(bar_token_id)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_dir", type=str, required=True)
    parser.add_argument("--inputs", type=str, required=True, help="prompt song")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated MIDI files")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--use_chord", action='store_true', help="Enable chord tokenization in the dataset")
    args = parser.parse_args()
    test(args)
