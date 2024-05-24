import numpy as np
import mdod

localFile = 'TestDataset.txt'
dets= np.loadtxt(localFile,delimiter=',')
nd = 1
sn = 10
result = mdod.md(dets,nd,sn)
print (result)
