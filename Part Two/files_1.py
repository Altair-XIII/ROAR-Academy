import os

# Define the file name and path
file_name = "motto.txt"
path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(path, file_name) # adds / when joining paths 

# Write to the file
try:
    with open(file_path, "w") as f_handle:
        f_handle.write("Fiat Lux!\n")
except IOError:
    print(f'Cannot open the file {file_name} for writing')
    exit()

# Read from the file
try:
    with open(file_path, "r") as f_handle:
        content = f_handle.read()
        print("File content after writing:")
        print(content)
except IOError:
    print(f'Cannot open the file {file_name} for reading')
    exit()

# Append to the file
try:
    with open(file_path, "a") as f_handle:
        f_handle.write("Let there be light!\n")
except IOError:
    print(f'Cannot open the file {file_name} for appending')
    exit()
