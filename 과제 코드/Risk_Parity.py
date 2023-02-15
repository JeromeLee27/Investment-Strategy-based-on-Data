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

#risk
risk1 = rawR1.rolling(window = 60).std()
risk2 = rawR2.rolling(window = 60).std()

#back test 시작일 해당 index
stDateNum = 19951215
stDate = pd.to_datetime(str(stDateNum), format = '%Y%m%d')

#Back Test 이후 데이터만
test_period = (rawTime>= stDate)
Time = rawTime[test_period].copy()
R1 = rawR1[test_period].copy()
R2 = rawR2[test_period].copy()
r1_ans = risk1[test_period].copy()
r2_ans = risk2[test_period].copy()
numData = R1.shape[0]

#포트폴리오 value, DD 계산
#weight 계산
w1 = risk2 / (risk1 + risk2)
w2 = 1 - w1
w1_ans = w1[test_period].copy()
w2_ans = w2[test_period].copy()

#portfoilo value
Rp = pd.Series(np.zeros(numData), index= Time)
Vp = pd.Series(np.zeros(numData), index= Time)

Vp[0] = 100
for t in range(1, numData):
    Rp[t] = w1_ans[t-1]*R1[t] + w2_ans[t-1]*R2[t]
    Vp[t] = Vp[t-1]*(1 + Rp[t]/100)
    
Maxp = Vp.cummax()
DDp = (Vp/Maxp - 1)*100

K200 = rawK200[rawK200.index >= stDate].copy()
Vb = (K200/K200[0])*100

#DD
MAXb = Vb.cummax()
DDb = (Vb/MAXb - 1)*100
#그래프 그리기
fig = plt.figure(figsize = (10,7))
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios = [8,3], width_ratios = [5])

ax0 = plt.subplot(gs[0])
ax0.plot(Time, Vp, label = 'Parity-Weight', color = 'red')
ax0.plot(Time, Vb, label = 'K200', color = 'blue')
ax0.set_title('<Value>')
ax0.grid(True)
ax0.legend()

ax1 = plt.subplot(gs[1])
ax1.plot(Time, DDp, color = 'red')
ax1.plot(Time, DDb, color = 'blue')
ax1.set_title('<Draw-down>')
ax1.grid(True)

plt.show()



