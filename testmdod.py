import numpy as np
import mdod

localFile = 'TestDataset.txt'
dets= np.loadtxt(localFile,delimiter=',')
# nd: value of the observation point in the new dimension
nd = 1
# sn: number of statistics on the first few numbers in the order of scores from large to small
sn = 10
result = mdod.md(dets,nd,sn)
print (result)
