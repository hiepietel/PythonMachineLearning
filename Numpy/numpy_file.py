import numpy as np
filedata = np.genfromtxt('data.txt', delimiter=",")
print(filedata)
filedata = filedata.astype('int32')
print(filedata)

print('bool and indexing')
print(filedata > 50)
print(filedata[filedata >50])

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[[1,2,8]])

filedata[filedata > 50] = 0
print(filedata)