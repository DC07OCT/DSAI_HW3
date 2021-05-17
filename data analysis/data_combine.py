#把資料整理在一起(得到generation_0507.csv,consumption_0507.csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

filepath = '/DSAI/training_data'
#0~49
data = pd.read_csv(filepath+'/target'+str(43)+'.csv')#43擁有最完整的time series
time_series=data['time']
#print(data['time'])#共5831項
# print(type(data['generation'][1]))#numpy.float64
#print(type(time_series[0]))#string
#print(time_series[0][:10])#取出日期"2018-01-01"
# print(time_series[0]==time_series[1])
# print(type((1,2,3,4,5)))#tuple
generation=[]
consumption=[]

#print('2018-01-01' in time_series[0]) #True
for i in range(50):
    data = pd.read_csv(filepath + '/target' + str(i) + '.csv')
    #print(data)
    temp1=[]
    temp1.append('target' + str(i))#加入head
    temp2=[]
    temp2.append('target' + str(i))
    k=0
    time_label=['Time']
    for j in range(len(time_series)):
        time_label.append(time_series[j])
        if data['time'][k]==time_series[j]:
            temp1.append(data['generation'][k])
            temp2.append(data['consumption'][k])
            k=k+1
        else:
            #temp1.append(float('NaN'))
            temp1.append(float(0))#放入數值創建 generation_0507.csv
            #temp2.append(float('NaN'))
            temp2.append(float(0))
    #print(k)
    if k!= len(data['time']):
        print('Error!')

    if i==0:
        generation.append(time_label)#time series標籤
        consumption.append(time_label)#time series標籤
    generation.append(temp1)
    consumption.append(temp2)

I=zip(*generation)
I1=zip(*consumption)#把資料從by column轉成 by row
T=list(I)
T1=list(I1)#轉換資料型態成[(tuple),(),()]

import csv

with open('generation_0507.csv','w', newline="") as g_file:#要寫generation 檔案
   writer = csv.writer(g_file)
   for row in range(len(T)):
       writer.writerow(list(T[row]))

with open('consumption_0507.csv','w', newline="") as c_file:#要寫consumption 檔案
   writer = csv.writer(c_file)
   for row in range(len(T1)):
       writer.writerow(list(T1[row]))

#從表格發現少data  2018/3/11  02:00:00 AM,已補齊
#共31+28+31+30+31+30+31+31=243->243*24=5832筆資料
