import os
import argparse
from midi2audio import FluidSynth

def convert_midi_to_wav(input_folder, output_folder, soundfont_path):
    """
    Converts all MIDI files in the input_folder to WAV files and saves them in the output_folder.

    Args:
        input_folder (str): Path to the folder containing MIDI files.
        output_folder (str): Path to the folder where WAV files will be saved.
        soundfont_path (str): Path to the .sf2 SoundFont file.

    Returns:
        None
    """
    # Initialize the FluidSynth object with the specified SoundFont
    fs = FluidSynth(soundfont_path)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all MIDI files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            midi_path = os.path.join(input_folder, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(output_folder, wav_filename)

            print(f"Converting {filename} to {wav_filename}...")

            # Convert MIDI to WAV
            fs.midi_to_audio(midi_path, wav_path)

            print(f"Saved: {wav_path}")

    print("All MIDI files have been converted to WAV format.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert MIDI files to WAV format.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing MIDI files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save WAV files.")
    parser.add_argument("--soundfont", type=str, required=True, help="Path to the .sf2 SoundFont file.")

    args = parser.parse_args()

    # Call the conversion function
    convert_midi_to_wav(args.input_folder, args.output_folder, args.soundfont)
