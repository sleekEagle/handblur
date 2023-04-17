import numpy as np
import matplotlib.pyplot as plt

f=60e-3
N=2.0
px=36*1e-6
s1=0.5

def get_blur_kcam(s2):
    return abs(s2-s1)/s2*1/(s1-f)*f**2/N/px

def get_blur(s2):
    return abs(s2-s1)/s2*1/(s1-f)

s2range=np.arange(0.5,1,0.001)

blurs=[]
blurs=[get_blur(s2) for s2 in s2range]

plt.plot(s2range,blurs)
plt.show()
max(blurs)

get_blur(0.97)

#blurs
f=50e-3 : 74.77
f=150e-3 : 865.24
