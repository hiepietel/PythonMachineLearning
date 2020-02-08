import numpy as np

a = np.array([1, 2, 3, 4])

print(a+2)
print(a-2)
print(a*2)
print(a/2)
print(a%2)
print(a**2)
print("a math")

b = np.array([1, 0, 1, 0])
print(a+b)

print(np.cos(a))

print("linear algebra")
c = np.ones((2, 3))
print(c)
d = np.full((3, 2), 2)
print(d)
print(np.matmul(c, d))
#https://docs.scipy.org/doc/numpy/reference/routines.linalg.html
e = np.identity(3)
print('det')
print(np.linalg.det(e))


print("statistics")
stats = np.array([[1,2,3],[4,5,6]])
print(np.max(stats))
print(np.sum(stats))

before = np.array([[1,2,3,4],[5,6,7,8]])
print(before.shape)

after = before.reshape((4,2))
print('after')
print(after)

print('vectors')
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

print(np.vstack([v1, v2]))


h1 = np.array([1, 2, 3, 4])
h2 = np.array([5, 6, 7, 8])

print(np.hstack((h1, h2)))
