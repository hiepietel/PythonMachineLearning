import matplotlib.pyplot as plt
import numpy as np
import random as rnd

timeDelta = 0.5 #5ms
n = 1000
kinh = np.random.ranf(n)
kinh = kinh % 0.2
kexc = np.random.ranf(n)
d = 8 * kexc + 2 * kinh
a = 0.02 * kexc + 0.1 * kinh

T = np.ceil(n/timeDelta)
v = np.zeros((n, int(T*timeDelta)))
v = v.astype('float')
u = np.zeros((n, int(T*timeDelta)))
u = u.astype('float')
v[:,0] = -70
u[:,0] = -14

nin = 1000
frate = 0.3 #[ms^(-1)]
taug = 10.0
gin = np.zeros((nin, 1))
Ein = np.zeros((nin, 1))
Win = np.random.rand(nin,n)
Win[Win > 0.1] = 0

for t in range(n-1):
    if t * timeDelta >150 and t* timeDelta < 350:
        p = np.random.ranf((nin,1))
        p[p>frate * timeDelta] = frate * timeDelta
    else:
        p = 0
    gin = gin + p
    ginEin = (gin * Ein)
    Iapp = Win * ginEin  #np.matmul(gin,Ein.reshape(1,nin))
    Iapp = Iapp - (Win*gin)
    Iapp = Iapp[:,t] * v[:,t]
    #print( Iapp*v[:,t])

    gin = (1 - timeDelta/taug) * gin
    dv = (0.04 * v[:, t] + 5 ) * v[:, t] + 140 - u[:,t]
    v[:, t+1] = v[:, t] + timeDelta * (dv +Iapp)#+Iapp[t,:])# + Iapp[:,t])
    du = a * (0.2 * v[:, t] - u[:, t])
    u[:, t+1] = u[:, t] + timeDelta * du

    for i in range(nin):
        if v[i, t] >= 35:
            v[i, t] = 35
            v[i, t+1] = -65
            u[i, t+1] =u[i,t] + d[i]


xtable = []
ytable = []
for i in range(v.shape[0]):
    for j in range(v.shape[1]):
        if v[i][j] == 35 :
          xtable.append(j)
          ytable.append(i)
plt.scatter(x = xtable, y = ytable, marker='o', edgecolors='r')
plt.title('Simulate 1000 neurons (not inter-connected)')
plt.xlabel('time [ms]')
plt.ylabel('uint [#]')
plt.ylim(0, n)
plt.xlim(0, n)
plt.show()
print(v.shape[0])
print(u.shape)
v[v<35] = 0
fig = plt.figure()


print(v[:,600])
#spks[kinh, :] = 2.0 * v[kinh, :]
#x = np.arange(0, n)
#y = np.arange(0, n)
#xx, yy = np.meshgrid(x, y)
#plt.contourf(xx, yy, v, cmap='jet')

#plt.colorbar()

