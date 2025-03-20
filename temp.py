import os

# Define the directory containing the files
directory = 'temps_EDF'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the filename contains 'B N,I'
    if 'B N,I ' in filename:
        # Create the new filename by removing 'B N,I '
        new_filename = filename.replace('B N,I ', '')
        # Get the full paths for the old and new filenames
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')