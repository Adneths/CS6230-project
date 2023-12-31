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


boffset = []
prefixsum_cuda = []
# boff_prefixsum = boffset
# for i in range(64*4):
#     for j in range(64):
#         if j != 0:
#             boff_prefixsum[j][i] += boff_prefixsum[j-1][i]
# print(np.where(boff_prefixsum!=prefixsum_cuda))
# print(boff_prefixsum)

# print(boff_prefixsum == prefixsum_cuda)
# appendix = []
# for i in range(64*4):
#     sum = 0
#     for j in range(64):
#         sum+= boffset[j][i]
#     appendix.append(sum)

# appendix = np.array(appendix)
# appendix_cuda = np.array(appendix_cuda)
# print(appendix)
# print(appendix == appendix_cuda)

boffset = np.array(boffset)

print(np.where(boffset))
# print(boffset[2][65])
# print(boffset[2][70])
# print(boffset[2][71])

colp = [0,6,9,14,17,30,38,48,70,70,70,70,74,74,81,84,85,85,85,100,102,104,105,105,106,107,107,107,107,110,111,113,115,115,115,115,115,119,122,126,128,129,130,130,132,132,132,132]
dataRow = [1,2,3,4,6,36,2,3,6,3,4,5,9,10,4,5,8,5,7,8,9,11,18,21,23,29,32,35,43,44,6,7,11,12,13,14,28,44,7,10,18,19,24,26,27,28,34,45,8,9,10,11,12,13,14,15,18,19,20,21,23,24,26,27,30,31,33,34,35,43,12,13,17,29,14,15,16,17,20,28,33,15,16,17,16,19,20,22,23,24,25,26,27,29,30,31,34,35,43,45,36,37,21,22,22,25,25,36,37,38,30,31,32,32,33,37,38,39,40,38,39,41,39,40,41,42,40,42,41,42,44,45]
for i in [8, 9, 11, 12, 20, 27, 29, 36, 39, 41, 45, 46]:
    print(i)
    cs = colp[i]
    ce = colp[i+1]
    data = dataRow[cs:ce]
    print(data)