import numpy as np

a = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])

arr = np.array([1,2,3])
r1 = np.repeat(arr, 2, axis=0)

out = np.ones((5, 5))
zero = np.zeros((3, 3))

zero[1, 1]= 9
out[1:-1,1:-1] = zero
print(out)

print(r1)

g = np.array([1,2,3])
h = g
h[0] = 100
print(g)
g[0] = 1
h = g.copy()
print(g)

#jedonstkowa
print(np.identity(3))
#random matrix
print(np.random.rand(2,3,4))
#rand int
print('rand int')
print(np.random.randint(7,size=(3,3)))
print(np.full_like(a.shape, 4))

print(np.full((2,2), 99, dtype='float'))
print(np.ones((4,2), dtype='int32'))
print(np.zeros((2,4)))
