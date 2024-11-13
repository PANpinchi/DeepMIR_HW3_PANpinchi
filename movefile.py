import os
import shutil

# Define the root directory
root_dir = './Pop1K7/midi_analyzed'

# List all subdirectories within the root directory
subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Move all MIDI files from subdirectories to the root directory
for subdir in subdirs:
    for file_name in os.listdir(subdir):
        if file_name.endswith('.mid'):
            source_path = os.path.join(subdir, file_name)
            dest_path = os.path.join(root_dir, file_name)

            # Move file
            shutil.move(source_path, dest_path)

    # Remove the subdirectory after moving files
    os.rmdir(subdir)

print("All MIDI files have been moved to", root_dir)
