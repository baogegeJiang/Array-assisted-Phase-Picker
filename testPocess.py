from multiprocessing import Process,Manager
import numpy as np

def setL(l,i):
    l.append(i)

manager=Manager()
l=[manager.list() for i in range(10)]
pL=[Process(target=setL, args=(l[i],i)) for i in range(10)]
for p in pL:
    p.start()


for p in pL:
    p.join()
print(l)

