# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:29:31 2022

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

data_path = './data_2assets.xlsx'
df_data = pd.read_excel(data_path, sheet_name = 'data_m', header = 1).set_index('Date')
df_data.head()

rawTime = df_data.index.copy()
rawK200 = df_data.iloc[:,0].copy()
rawUSD = df_data.iloc[:,4].copy()
rawLIBOR1m = df_data.iloc[:, 5].copy()

#C/G HPR
rawCR1 = rawK200.pct_change()*100
rawCR2 = rawUSD.pct_change()*100

#I/G HPR
rawIR1 = 0
rawIR2 = rawLIBOR1m.shift(1)/12

#HPR = C/G + I/G
rawR1 = rawCR1 + rawIR1
rawR2 = ((1 + rawCR2/100)*(1+rawIR2/100)-1)*100

#back test 시작일 해당 index
stDateNum = 19951215
stDate = pd.to_datetime(str(stDateNum), format = '%Y%m%d')

#Back Test 이후 데이터만
test_period = (rawTime>= stDate)
Time = rawTime[test_period].copy()
R1 = rawR1[test_period].copy()
R2 = rawR2[test_period].copy()
numData = R1.shape[0]

#포트폴리오 value, DD 계산
#weight
w1_1 = 0.5 ; w1_2 = 1 - w1_1

#risk
risk1 = rawR1.rolling(window = 60).std()
risk2 = rawR2.rolling(window = 60).std()
w2_1 = risk2 / (risk1 + risk2)
w2_2 = 1 - w2_1
w2_1_ans = w2_1[test_period].copy()
w2_2_ans = w2_2[test_period].copy()
w2_1
w2_1_ans
w2_2_ans


cov12 = rawR1.rolling(60).cov(rawR2)
risk3_1 = (risk2**2 - cov12) / (risk1**2+ risk2**2 - 2*cov12)
risk3_2 = 1 - risk3_1

w3_1_ans = risk3_1[test_period].copy()
w3_2_ans = risk3_2[test_period].copy()


#portfoilo value
Rp1 = pd.Series(np.zeros(numData), index= Time)
Vp1 = pd.Series(np.zeros(numData), index= Time)

Rp1

R1.shape[0]

Rp2 = pd.Series(np.zeros(numData), index= Time)
Vp2 = pd.Series(np.zeros(numData), index= Time)

Rp3 = pd.Series(np.zeros(numData), index= Time)
Vp3 = pd.Series(np.zeros(numData), index= Time)

Vp1[0] = 100 ; Vp2[0] = 100 ; Vp3[0] = 100

R1
#equal-weighted
for t in range(1, numData):
    Rp1[t] = w1_1*R1[t] + w1_2*R2[t]
    Vp1[t] = Vp1[t-1]*(1 + Rp1[t]/100)
    
Maxp1 = Vp1.cummax()
DDp1 = (Vp1/Maxp1 - 1)*100

Rp2

w2_1_ans
w2_2_ans

#risk-parity
for t in range(1, numData):
    Rp2[t] = w2_1_ans[t-1]*R1[t] + w2_2_ans[t-1]*R2[t]
    Vp2[t] = Vp2[t-1]*(1 + Rp2[t]/100)
    
Maxp2 = Vp2.cummax()
DDp2 = (Vp2/Maxp2 - 1)*100

#MVP
for t in range(1, numData):
    Rp3[t] = w3_1_ans[t-1]*R1[t] + w3_2_ans[t-1]*R2[t]
    Vp3[t] = Vp3[t-1]*(1 + Rp3[t]/100)

Maxp3 = Vp3.cummax()
DDp3 = (Vp3/Maxp3 - 1)*100

K200 = rawK200[rawK200.index >= stDate].copy()
Vb = (K200/K200[0])*100

MAXb = Vb.cummax()
DDb = (Vb/MAXb - 1)*100
#그래프 그리기
fig = plt.figure(figsize = (10,7))
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios = [8,3], width_ratios = [5])

ax0 = plt.subplot(gs[0])
ax0.plot(Time, Vp1, label = 'Fixed-Weight', color = 'red')
ax0.plot(Time, Vp2, label = 'Equal-Parity', color = 'blue')
ax0.plot(Time, Vp3, label = 'MVP', color = 'green')
ax0.plot(Time, Vb, label = 'K200', color = 'black')
ax0.set_title('<Value>')
ax0.grid(True)
ax0.legend()

ax1 = plt.subplot(gs[1])
ax1.plot(Time, DDp1, label = 'Fixed-Weight', color = 'red')
ax1.plot(Time, DDp2, label = 'Equal-Parity', color = 'blue')
ax1.plot(Time, DDp3, label = 'MVP', color = 'green')
ax1.plot(Time, DDb, label = 'K200', color = 'black')
ax1.set_title('<Draw-down>')
ax1.grid(True)

plt.show()

# Annual Performance
rf = df_data.iloc[:, 6].copy()
rf_na = rf.dropna()

rf_value = pd.Series(np.zeros(Vb.iloc[::12,].shape[0]), index= Annual.index)
rf_value[0] = 100
for t in range(1, rf_value.shape[0]):
    rf_value[t] = rf_value[t-1] *(rf_na[t-1]/100 + 1)

Vp_all = pd.concat([Vb, Vp1, Vp2, Vp3], axis = 1)
Vp_all
Annual_Vp = Vp_all.iloc[::12,]
Annual_Rp = 100* (Annual_Vp / Annual_Vp.shift(1) - 1)
Annual_Vp.insert(1,'RF',rf_value)
Annual_Rp.insert(1,'RF_rate',rf_na)

Annual = pd.concat([Annual_Vp, Annual_Rp], axis = 1)
Annual.columns = ['K200','RF', 'EW', 'RP', 'MV','K200_rate', 'RF_rate', 'EW_rate', 'RP_rate', 'MV_rate']
Annual.head()
Annual

#Annual Performance Evaluation
df = pd.DataFrame(index=['K200', 'RF', 'EW', 'RP', 'MV'], columns = ['V_T', 'CAGR', 'MDD', 'E_R', 'sigma_R', 'CV', 'sharpe'])

df.iloc[[0,2,3,4],0] =  (Vp_all.iloc[-1,]).values
df.iloc[1,0] = Annual.iloc[-1,1]
df['CAGR'] = 100 * ((Annual_Vp.iloc[-1,] / Annual_Vp.iloc[0,]) ** (1/len(Annual_Vp)) - 1).values
MDD = pd.concat([DDb, DDp1, DDp2, DDp3], axis = 1).min()
df.iloc[[0,2,3,4],2] = MDD.values
df['E_R'] = Annual_Rp.mean().values
df['sigma_R'] = Annual_Rp.std().values
df['CV'] = (sigma_R / E_R).values
df.iloc[[0,2,3,4],6] = ((E_R[['K200', 0,1,2]] - E_R['RF_rate']) / sigma_R[['K200', 0,1,2]]).values
annual_eva = df.T
annual_eva