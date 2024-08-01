input_file = "groundTruth/60fps/014_Sebastien_ter.txt"
output_file = "groundTruth/60fps/014_Sebastien_ter_.txt"

with open(input_file, 'r') as file:
    lines = file.readlines()

modified_lines = [line.rstrip().rstrip('_S') + '\n' if line.rstrip().endswith('_S') else line for line in lines]

with open(output_file, 'w') as file:
    file.writelines(modified_lines)