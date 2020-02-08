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
v = np.zeros((n, int(T)))
v = v.astype('float')
u = np.zeros((n, int(T)))
u = u.astype('float')
v[:,0] = -70
u[:,0] = -14

nin = 100
frate = 0.002 #[ms^(-1)]
taug = 10.0
gin = np.zeros((nin, 1))
Ein = np.zeros((nin, 1))
Win = np.random.rand(nin,n)
Win[Win >0.1] = 0

for t in range((int)(n-1)):
    if t * timeDelta > 200 and t * timeDelta < 700:
        p = np.random.ranf((nin,1))
        p = p % frate * timeDelta
    else:
        p = 0
    gin = gin + p
    ginEin = (gin * Ein)
    Iapp = Win *ginEin  #np.matmul(gin,Ein.reshape(1,nin))
    Iapp = Iapp - (Win*gin)
    #print( Iapp*v[:,t])

    gin = (1 - timeDelta/taug) * gin
    dv = (0.04 * v[:,t] +5) * v[:,t] +140 - u[:,t]
    v[:,t+1] = v[:, t] + timeDelta * (dv)#+Iapp[t,:])# + Iapp[:,t])
    du = a * (0.2 * v[:,t] - u[:,t])
    u[:,t+1] = (u[:, t] + timeDelta * du)

    fired = v[:, t]
    for i in range(nin):
        if fired[i] > 35:
            v[i, t] = 35
            v[i, t+1] = -65
            u[i, t+1] =u[i,t] + d[i]
            print("fired")


    print(t)

print(v.shape)
print(u.shape)
plt.matshow(v)
print(u[:,300])
#spks[kinh, :] = 2.0 * v[kinh, :]
x = np.arange(0, T)
y = np.arange(0, n)
xx, yy = np.meshgrid(x, y)
plt.contourf(xx, yy, u, cmap='jet')

plt.colorbar()
plt.show()

