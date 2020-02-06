#https://www.youtube.com/watch?v=QUT1VHiLmmI
import numpy as np
a = np.array([1, 2, 3])
print(a)

b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
print('dimnesional ', b.ndim)
print('shape ', b.shape)
print('dtype ', b.dtype)
print("Get Size: ", b.itemsize)
print('Get total Size', b.nbytes)

c = np.array([[1, 2, 3, 4, 5, 6, 7],
              [8, 9, 10, 11, 12, 13, 14]])
#get element
print(c[1,5])
#get specyfic row
print(c[0, :])
print(c[:, 2])
#get [dstartindex:endindex:stepsize]
print(c[:, 1:6:2])

d = np.array([[[1,2], [3,4],[5,6],[7,8]]])
print(d)

print(d[:,1,:])
#d[:]
