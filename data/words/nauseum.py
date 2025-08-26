import re
import os
import sys

# Get the directory path from user input
directory = sys.argv[1]

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # Process only text files
        file_path = os.path.join(directory, filename)
        
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            text = file.read()
        
        # Count the words
        word_count = len(re.findall(r'\b\w+\b', text))
        
	print("File: {}, Total words: {}".format(filename, word_count))
