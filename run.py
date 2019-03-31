import os
import time
import serial
import warnings
import numpy as np
import csv


port = ''
#s = serial.Serial(port,9600)

os.system('python3 -W ignore demo_cpm_hand.py --DEMO_TYPE test_imgs/hand.jpg')
v=[]
with open('test_file.csv') as file:
	reader = csv.reader(file, delimiter=',')
	for row in reader:
		v.append(row[0])
		v.append(row[1])



'''
f=open("testfile.txt","r")
v=f.readline()

print('running...')
v = v.strip().split(' ')
'''

ratio_n = []
x1 = (float(v[0])+float(v[2])+float(v[6])+float(v[10])+float(v[14]))/5
y1 = (float(v[1])+float(v[3])+float(v[7])+float(v[11])+float(v[15]))/5


for i in range(4):
	x2 = float(v[4*i+2])
	y2 = float(v[4*i+3])
	x3 = float(v[4*i+4])
	y3 = float(v[4*i+5])

	fullen1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
	fullen2 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
	ratio_n.append(fullen2) 
	#ratio_n.append(fullen2/fullen1) 

#f.close()
print(ratio_n)


'''
f=open("testfile.txt","w")
for x in ratio_n:
	f.write(str(x)+' ')
f.close()

os.system('python3 -W ignore demo_cpm_hand.py --DEMO_TYPE SINGLE')
'''

with open('test_file.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(ratio_n)
os.system('python3 -W ignore demo_cpm_hand.py --DEMO_TYPE SINGLE')
'''
for i in ['second','third','fourth','fifth']:
	ratio = []
	os.system('python3 -W ignore demo_cpm_hand.py --DEMO_TYPE test_imgs/'+i+'.jpg')
	f=open("testfile.txt","r")
	v=f.readline()
	v = v.strip().split(' ')

	x1 = float(v[0])
	y1 = float(v[1])
	
	for j in range(4):
		x2 = float(v[4*j+2])
		y2 = float(v[4*j+3])
		x3 = float(v[4*j+4])
		y3 = float(v[4*j+5])

		length1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
		length2 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
		ratio.append(length2/length1)

	ratio = np.divide(np.array(ratio),np.array(ratio_n))
	ratio=ratio*(ratio<1)+(ratio>=1)
	ratio=ratio*(ratio>-1)-(ratio<=-1)

	deg = np.degrees(np.arccos(ratio))
	print(deg)
	#s.write(str(deg).encode())
	f.close()
'''	