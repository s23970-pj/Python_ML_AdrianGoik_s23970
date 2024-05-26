# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:51:11 2020

@author: Tomek
"""
a=123
b=321
print(a*b)
print(a+b)

import numpy as np
c = np.array([3,8,9,10,12])
d = np.array([8,7,7,5,6])
print(c+d)
print(c*d)
print(np.dot(c,d))
print(np.linalg.norm(d-c))

A = np.array([[1,4,7],[2,5,8],[3,6,9]])
B = A
print(A)
print(B)
print(A*B)
print(np.dot(A,B))
print(100*np.random.rand(1,50))
print(np.random.randint(1,100, (1,50)))
C = np.random.randint(1,100, (1,3))
print(C)
print(C.mean())
print(C.max())
print(C.min())
print(C.std())

def normalize(x):
    return((x-x.min())/(x.max()-x.min()))    
print(normalize(C))

import csv
with open('LAB1/miasta.csv','a', newline='') as newFile:
    newFileWriter=csv.writer(newFile)
    newFileWriter.writerow([2010,460,555,405])

x = []
y = []
z = []
t = []

with open('LAB1/miasta.csv','r') as newFile:
    plots = csv.reader(newFile, delimiter=',')
    has_header = csv.Sniffer().has_header(newFile.read(1024))
    newFile.seek(0) #na poczatek
    if has_header:
        next(plots) 
    for row in plots:
        print(row)
        x.append(int(row[0]))
        y.append(int(row[1]))
        z.append(int(row[2]))
        t.append(int(row[3]))
    
import matplotlib.pyplot as plt

plt.plot(x,y, color='r')
plt.plot(x,z, color='b')
plt.plot(x,t, color='y')
plt.legend(('Gdansk', 'Poznan', 'Szczecin'))
plt.xlabel('Lata')
plt.ylabel('Liczba ludnosci')
plt.title('Ludnosc w miastach Polski')
plt.show()




 







