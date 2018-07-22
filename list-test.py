import numpy as np

mydict = {}
mylist0 = np.array([1, 2, 3, 4, 5])
mylist1 = np.array([2, 3, 4, 5, 6])

print(mydict)
print(mylist0)
print(mylist1)

for c in ('0', '1'):
    if c in mydict:
        mydict[c] += mylist0
    else:
        mydict[c] = mylist0

print(mydict)

for c in ('0', '1'):
    if c in mydict:
        mydict[c] += mylist1
    else:
        mydict[c] = mylist1

print(mydict)

