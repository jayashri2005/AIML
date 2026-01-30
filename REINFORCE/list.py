l1=['a','b','c','d']
l2=[10,20,30,40]
print('  '.join(str(i) for i in l1))
print(' '.join(str(i) for i in l2))

#using zip
print()
for i,j in zip(l1,l2):
    print(i,j)
