# DeepMIR HW3: Symbolic Music Generation

## Overview
* Learn to manipulate MIDI file and represent symbolic music as tokens. 
* Learn to train a transformer-based model for symbolic music generation.


### Task 1: Symbolic music generation 
Train a transformer-based model from scratch to generate 32 bars symbolic music.
* You may randomly select the prompt, treat it as a priming sequence for Transformer decoder to generate a continuation
    * For example: [Bar_None, Position_1/16, Tempo_class, Tempo_value]
* Model: You can use any transformer-based model.
* Either 1-stage generation or 2-stage generation is fine.

### Task 2: Symbolic music continuation
TA will provide 3 midi files (8 bars) as prompt, you need to generate their continuation for 24 bars. (Total: 8+24 bars)
* You can use checkpoint in Task 1, don’t need to train another model.

### Dataset: Pop1K7
* 1747 pop music piano performances (mid or midi file) transcribed from youtube audio.
* Single track, 4 minutes in duration, totaling 108 hours.   
* 4/4 time signature (four beats per bar).

### Evaluation
Objective metrics for generation results
* Pitch-Class Histogram Entropy (H) : measures erraticity of pitch usage in shorter timescales (e.g., 1 or 4 bars). (H4 is required)
* Grooving Pattern Similarity (GS) : measures consistency of rhythm across the entire piece. (required)
* Structureness Indicator (SI) : detects presence of repeated structures within a specified range of timescale. (optional)


## Getting Started 
```bash
# Clone the repo:
git clone https://github.com/PANpinchi/DeepMIR_HW3_PANpinchi.git
# Move into the root directory:
cd DeepMIR_HW3_PANpinchi
```
## Environment Settings
```bash
# Create a virtual conda environment:
conda create -n deepmir_hw3 python=3.10

# Activate the environment:
conda activate deepmir_hw3

# Install PyTorch, TorchVision, and Torchaudio with CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Download GPT2RGA.py
wget 'https://raw.githubusercontent.com/asigalov61/tegridy-tools/main/tegridy-tools/GPT2RGA.py'

# Install additional dependencies from requirements.txt:
pip install -r requirements.txt
```

## Download the Required Data
#### 1. Pre-trained Models
Run the commands below to download the pre-trained model.
```bash
# The pre-trained GPT-2 model with Chord
gdown --folder https://drive.google.com/drive/folders/1pUOR4So048nR02UIk4no-KxgkU00oBn4?usp=drive_link

# The pre-trained GPT-2 model without Chord
gdown --folder https://drive.google.com/drive/folders/1EdvkETVFvjD-dDXKI94imOfndh06S0Yu?usp=drive_link
```
Note: The `best_model.pth` files should be placed in the `setting_1` and `setting_2` folders respectively.

#### 2. Datasets
You need to execute the following commands to download, unzip and preprocess Pop1K7 dataset.
```bash
# Download the Pop1K7 dataset
wget -O Pop1K7.zip "https://zenodo.org/records/13167761/files/Pop1K7.zip?download=1"

# Unzip
unzip Pop1K7.zip

# Preprocess
python movefile.py
```

## 【Task 1: Symbolic music generation】
#### Training
```bash
# Train a GPT-2 model with Chord.
python main.py --checkpoint_folder setting_1 --batch_size 8 --use_chord

# Train a GPT-2 model without Chord.
python main.py --checkpoint_folder setting_2 --batch_size 8
```

#### Inference
```bash
# Settting 1: Test a GPT-2 model with Chord.
python generate_32_bars.py --use_chord --saved_dir setting_1 --top_k 50 --temperature 1.0 --output_dir outputs_1

# Settting 2: Test a GPT-2 model with Chord.
python generate_32_bars.py --use_chord --saved_dir setting_1 --top_k 50 --temperature 2.0 --output_dir outputs_2

# Settting 3: Test a GPT-2 model with Chord.
python generate_32_bars.py --use_chord --saved_dir setting_1 --top_k 100 --temperature 1.0 --output_dir outputs_3

# Settting 4: Test a GPT-2 model without Chord.
python generate_32_bars.py --saved_dir setting_2 --top_k 50 --temperature 1.0 --output_dir outputs_4

# Settting 5: Test a GPT-2 model without Chord.
python generate_32_bars.py --saved_dir setting_2 --top_k 50 --temperature 2.0 --output_dir outputs_5

# Settting 6: Test a GPT-2 model without Chord.
python generate_32_bars.py --saved_dir setting_2 --top_k 100 --temperature 1.0 --output_dir outputs_6
```

#### Evaluation
```bash
# Compute metrics on real data
python eval_metrics.py --dict_path basic_event_dictionary.pkl --output_file_path Pop1K7/midi_analyzed --output_csv pop1k7.csv

# Compute metrics on Setting 1 results
python eval_metrics.py --dict_path basic_event_dictionary.pkl --output_file_path outputs_1 --output_csv outputs_1/outputs_1.csv

# Compute metrics on Setting 2 results
python eval_metrics.py --dict_path basic_event_dictionary.pkl --output_file_path outputs_2 --output_csv outputs_2/outputs_2.csv

# Compute metrics on Setting 3 results
python eval_metrics.py --dict_path basic_event_dictionary.pkl --output_file_path outputs_3 --output_csv outputs_3/outputs_3.csv

# Compute metrics on Setting 4 results
python eval_metrics.py --dict_path basic_event_dictionary.pkl --output_file_path outputs_4 --output_csv outputs_4/outputs_4.csv

# Compute metrics on Setting 5 results
python eval_metrics.py --dict_path basic_event_dictionary.pkl --output_file_path outputs_5 --output_csv outputs_5/outputs_5.csv

# Compute metrics on Setting 6 results
python eval_metrics.py --dict_path basic_event_dictionary.pkl --output_file_path outputs_6 --output_csv outputs_6/outputs_6.csv
```

## 【Task 2: Symbolic music continuation】
#### Inference
```bash
# Settting 1: Test a GPT-2 model without Chord.
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_1.mid --top_k 50 --temperature 1.0 --output_dir outputs_1
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_2.mid --top_k 50 --temperature 1.0 --output_dir outputs_1
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_3.mid --top_k 50 --temperature 1.0 --output_dir outputs_1

# Settting 2: Test a GPT-2 model without Chord.
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_1.mid --top_k 50 --temperature 2.0 --output_dir outputs_2
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_2.mid --top_k 50 --temperature 2.0 --output_dir outputs_2
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_3.mid --top_k 50 --temperature 2.0 --output_dir outputs_2

# Settting 3: Test a GPT-2 model without Chord.
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_1.mid --top_k 100 --temperature 1.0 --output_dir outputs_3
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_2.mid --top_k 100 --temperature 1.0 --output_dir outputs_3
python inference.py --use_chord --saved_dir setting_1 --inputs prompt_song/song_3.mid --top_k 100 --temperature 1.0 --output_dir outputs_3

# Settting 4: Test a GPT-2 model without Chord.
python inference.py --saved_dir setting_2 --inputs prompt_song/song_1.mid --top_k 50 --temperature 1.0 --output_dir outputs_4
python inference.py --saved_dir setting_2 --inputs prompt_song/song_2.mid --top_k 50 --temperature 1.0 --output_dir outputs_4
python inference.py --saved_dir setting_2 --inputs prompt_song/song_3.mid --top_k 50 --temperature 1.0 --output_dir outputs_4

# Settting 5: Test a GPT-2 model without Chord.
python inference.py --saved_dir setting_2 --inputs prompt_song/song_1.mid --top_k 50 --temperature 2.0 --output_dir outputs_5
python inference.py --saved_dir setting_2 --inputs prompt_song/song_2.mid --top_k 50 --temperature 2.0 --output_dir outputs_5
python inference.py --saved_dir setting_2 --inputs prompt_song/song_3.mid --top_k 50 --temperature 2.0 --output_dir outputs_5

# Settting 6: Test a GPT-2 model without Chord.
python inference.py --saved_dir setting_2 --inputs prompt_song/song_1.mid --top_k 100 --temperature 1.0 --output_dir outputs_6
python inference.py --saved_dir setting_2 --inputs prompt_song/song_2.mid --top_k 100 --temperature 1.0 --output_dir outputs_6
python inference.py --saved_dir setting_2 --inputs prompt_song/song_3.mid --top_k 100 --temperature 1.0 --output_dir outputs_6
```

