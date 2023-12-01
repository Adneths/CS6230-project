import numpy as np

file = "results/GD.txt"

i = 0
mat = []
vec = []
matdriven_vec = []
vecdriven_vec = []

with open(file, 'r') as f:
    for line in f:
        i += 1
        if i <5:
            continue
        elif i == 5:
            nrow = int(line.split(' ')[1].split('\n')[0])
            print(nrow)
        elif i > 11 and i < 12 + nrow:
            row = line.split('\n')[0].split('\t')[:-1]
            row = [int(j) for j in row]
            mat.append(row)
        elif i == 12 + nrow + 3:
            row = line.split('\n')[0].split('\t')[:-1]
            vec = [int(j) for j in row]
        elif i == 12 + nrow + 3 + 4:
            row = line.split('\n')[0].split('\t')[:-1]
            matdriven_vec = [int(j) for j in row]
        elif i == 12 + nrow + 3 + 4 + 4:
            row = line.split('\n')[0].split('\t')[:-1]
            vecdriven_vec = [int(j) for j in row]  
mat = np.array(mat)
vec = np.array(vec)
matdriven_vec = np.array(matdriven_vec)
vecdriven_vec = np.array(vecdriven_vec)

correct = np.dot(mat, vec)
print("correct == matdriven: {}".format(np.all(correct == matdriven_vec)))
print("correct == vecdriven: {}".format(np.all(correct == vecdriven_vec)))
