#라이브러리 로드
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings; warnings.simplefilter('ignore')

import statsmodels.formula.api as smf
os.chdir(r'D:\바탕화면\2022 - 2\데이터기반투자전략\개별종목WML')

#데이터 로드
#기존에 수업시간에 이용했던 데이터들을 이용한다.
price = pd.read_csv('코스피+코스닥(상폐포함)_수정주가.csv', index_col = 'Date')
rawMarket = pd.read_csv('코스피+코스닥(상폐포함)_상장시장정보.csv', index_col= 'Date')
rawSector = pd.read_csv('코스피+코스닥(상폐포함)_Sector.csv', index_col = 'Symbol')
rawFactor = pd.read_csv('FatorReturn(KOSPI).csv', index_col = 'Date')

rawBM = rawFactor['MKT'].copy()
rawRF = rawFactor['RF'].copy()

#ESG등급 표에는 크롤링한 데이터를 이용.
esg = pd.read_excel('ESG기업등급표.xlsx')

#문자형 변경
esg['코드'] = esg['코드'].apply(str)

#esg기업등급표와 원래의 각 종목별 코드가 맞지 않아 맞춰주는 작업임. 
for i in range(esg.shape[0]):
    if len(esg['코드'][i]) < 6 :
        esg['코드'][i] = 'A' + '0' * (6-len(esg['코드'][i])) + esg['코드'][i] 
    else :
        esg['코드'][i] = 'A' + esg['코드'][i]

#ESG등급 A+, A, B+를 long으로 잡고 / ESG등급 C, D를 short으로 잡고 계산한 방식
win = esg.loc[(esg['ESG등급'] == 'A+') | (esg['ESG등급'] == 'A') | (esg['ESG등급'] == 'B+')]
lose = esg.loc[(esg['ESG등급'] == 'C') | (esg['ESG등급'] == 'D')]

win_esg = win['코드'] ; lose_esg = lose['코드']

#price와 ESG등급코드를 맞춰주기
price = price.T
price = price.reset_index()  
price.rename(columns = {'index' : '코드'}, inplace = True)
price.head()

combined = pd.merge(esg, price,  how = 'inner', on = '코드')

#win에 ESG등급 A+, A, B+ / lose에 ESG등급 C, D를 잡음. 
win = combined.loc[(combined['ESG등급'] == 'A+') | (combined['ESG등급'] == 'A') | (combined['ESG등급'] == 'B+')]
lose = combined.loc[(combined['ESG등급'] == 'C') | (combined['ESG등급'] == 'D')]
win.drop(['Unnamed: 0','기업명', 'ESG등급', '환경', '사회', '지배구조' ], axis = 1, inplace = True)
lose.drop(['Unnamed: 0','기업명', 'ESG등급', '환경', '사회', '지배구조' ], axis = 1, inplace = True)
win = win.T ; lose = lose.T
win.columns = win.iloc[0, :] 
lose.columns = lose.iloc[0, :]
win.drop('코드', axis = 0, inplace = True)
lose.drop('코드', axis = 0, inplace = True)

win.shape, lose.shape

#계산 편하게 하기 위해 datetime변경
win.index = pd.to_datetime(win.index)
lose.index = pd.to_datetime(lose.index)

#비교군인 BM와 RF데이터도 정리
rawBM.index = pd.to_datetime(rawBM.index)
rawRF.index = pd.to_datetime(rawRF.index)

#월별 데이터 추출
win = win.resample('BM').last().copy()
lose = lose.resample('BM').last().copy()

BM = rawBM.resample('BM').last().copy()
MonthRF = rawRF.resample('BM').last().copy()

#월별 수익률 확인하는 작업
RF = MonthRF.shift(1)/12 
win_r = (win.diff() / win.shift(1)) * 100
lose_r = (lose.diff() / lose.shift(1)) * 100
bmReturn = (BM.diff() / BM.shift(1))*100

win_r.shape, BM.shape, RF.shape, lose_r.shape
Time = win.index 

#back test
#기준을 2010년 데이터로 잡았기 때문에 해당 데이터로만 날짜 뽑아내는 방식
stDateNum = 20100101
stDate = pd.to_datetime(str(stDateNum), format = '%Y%m%d')

win_r = win_r[win_r.index >= stDate].copy()
lose_r = lose_r[lose_r.index >= stDate].copy()

Time = Time[Time>= stDate].copy()
bmReturn = bmReturn[bmReturn.index >= stDate].copy()
RF = RF[RF.index >= stDate].copy()

#-----------------------------------------------------------------
#여기까지 코드들은 'ESG_Equal_Weighted방식코드.py'의 윗부분 내용과 동일한 코드이다.
#-----------------------------------------------------------------

#곱하는 weight를 시총규모에 맞춰 값을 달리 해주기 위해서 각 종목에 맞는 시총규모 데이터를 로드한다.
size_win = pd.read_excel('시총규모.xlsx', sheet_name = 'Long 시총', index_col = 'Date')
size_lose = pd.read_excel('시총규모_re.xlsx', sheet_name = 'Short 시총', index_col = 'Date')

win_group = win_r.copy()
win_weight = size_win.copy()
lose_group = lose_r.copy()
lose_weight = size_lose.copy()

#복사된 시총 데이터에서 nan값으로 되어 있는 건 계산을 원활하게 하기 위해 0으로 바꿔준다.
win_weight.replace(0, np.nan, inplace = True)
lose_weight.replace(0, np.nan, inplace = True)

wina = win_weight.copy()
losea = lose_weight.copy()

#해당 시기에 존재하는 종목들의 시가총액 합을 구한 뒤, 이에 근거해서 (각 종목의 시가총액 / 전체 종목 시가총액)으로 weight를 맞춰준다.
for t in tqdm(Time):
    size = win_weight.loc[t].sum()
    wina.loc[t] = win_weight.loc[t] / size

for t in tqdm(Time):
    size = lose_weight.loc[t].sum()
    losea.loc[t] = lose_weight.loc[t] / size

wina = wina.shift(1) ; losea = losea.shift(1)

#win, lose포지션별로 수익률 데이터 합해주기
df_Return = pd.DataFrame(columns = ['ESG_win', 'ESG_lose'], index = Time)

#각 종목별 sum(해당 월 수익률 * 해당 월 전체 중 비율)해서 월별 데이터로 뽑아내기 과정 
win_accum = (win_group * wina) ; lose_accum = (lose_group * losea)

df_Return['ESG_win'] = win_accum.sum(axis = 1)
df_Return['ESG_lose'] = lose_accum.sum(axis = 1)
df_Return['win_lose'] = df_Return['ESG_win'] - df_Return['ESG_lose']

df_Value = (1+df_Return/100).cumprod()*100 #초기 투자를 100으로 잡고 누적곱해준 것
df_Maxp = df_Value.cummax()
df_DDp = (df_Value/df_Maxp -1) * 100    

#win long, lose short을 둘다 진행해 구한 수익률, 가격
Rp1 = df_Return.iloc[:,0] - df_Return.iloc[:,-1]
Vp1 = (1+Rp1/100).cumprod() *100 ; 
Maxp1 = Vp1.cummax()
DDp1 = (Vp1/Maxp1 - 1)*100

#win long을 진행해 구한 수익률, 가격
Rp2 = df_Return.iloc[:,0]
Vp2 = (1+Rp2/100).cumprod() *100 ; 
Maxp2 = Vp2.cummax()
DDp2 = (Vp2/Maxp2 - 1)*100

#--------------------------------------------------------------------------------
#팩터회귀분석을 위한 전처리과정 / 원래는 팩터회귀분석을 할 때 이를 시행해주는 것이 좋지만,
#Win long, MKF2000 short한 것을 가지고 Rp, Vp를 구하기 위해서 전처리 과정 지금 진행
#--------------------------------------------------------------------------------

#Factor Regression
rawFactor = pd.read_excel('FactorReturn.xlsx', sheet_name = 'data', header = 2).set_index('Date')
rawFactor.index = pd.to_datetime(rawFactor.index)

#월별 데이터 추출
Factor = rawFactor.resample('BM').last().copy()
FactorRF = Factor['RF'].shift(1)/12 #riskfree 수익률 계산
FactorReturn = Factor.drop(['RF'], axis = 1).pct_change()*100

#MKR(Market - RF) 산출
FactorReturn['MKT'] = FactorReturn['MKF2000'] - FactorRF

#Back Test 시작일과 factor data 기간 일치.
FactorReturn = FactorReturn[FactorReturn.index >= stDate].copy()

#WML 포함된 dataframe 생성
FactorReg = pd.concat([FactorReturn, df_Return], axis = 1)
FactorReg['ESG_MKT'] = FactorReg['win_lose'] - FactorReg['MKF2000']

#22년 9월 데이터까지 있기 때문에 이를 맞춰주기 위해서 다음과 같이 slicing진행
FactorReg = FactorReg.iloc[:-9, ]

#win long, MKF2000 short을 진행해 구한 수익률, 가격 
#바로 위에서 FactorReg dataframe의 'ESG_MKT'변수
#FactorReg['ESG_MKT'] = FactorReg['win_lose'] - FactorReg['MKF2000']
#에서 이용한 걸 그대로 사용

#ESG - MKF2000
Rp3 = FactorReg['ESG_MKT']
Vp3 = (1+Rp3/100).cumprod() *100 ; 
Maxp3 = Vp3.cummax()
DDp3 = (Vp3/Maxp3 - 1)*100

#Benchmark 수익률
Rp_bm = bmReturn.copy()
Rp_bm = Rp_bm - RF
Vp_bm = (1+Rp_bm.shift(-1)/100).cumprod() * 100
Vp_bm = Vp_bm.shift(1) ; Vp_bm[0] = 100 
Maxp_bm = Vp_bm.cummax()
DDp_bm = (Vp_bm / Maxp_bm -1) * 100

#그래프 그리기
def group_show(data1, data2, data3, data4):
    fig = plt.figure(figsize = (10,7))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios = [8,3], width_ratios = [5])
    
    ax0 = plt.subplot(gs[0])
    ax0.plot(Time, data1, label = 'Portfolio', color = 'red')
    ax0.plot(Time, data2, label = 'Benchmark', color = 'blue')
    ax0.set_title('<Value>')
    ax0.grid(True)
    ax0.legend()
    
    ax1 = plt.subplot(gs[1])
    ax1.plot(Time, data3, label = 'Portfolio', color = 'red')
    ax1.plot(Time, data4, label = 'Benchmark', color = 'blue')
    ax1.set_title('<Draw-down>')
    ax1.grid(True)
    
    plt.show()


group_show(Vp1, Vp_bm, DDp1, DDp_bm) #win long - lose short
group_show(Vp2, Vp_bm, DDp2, DDp_bm) #win long
group_show(Vp3, Vp_bm, DDp3, DDp_bm) #win long - mkf2000 short

#--------------------------------------------------------------------------------
#ESG팩터에 대한 팩터회귀분석과 그 결과에 대한 해석
#--------------------------------------------------------------------------------

# 3 - Factor Regression OLS
#win-lose ~ 3 factor
model = smf.ols(formula = 'win_lose ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())


#win ~ 3 factor
model = smf.ols(formula = 'ESG_win ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())

#win-lose long - mkf2000 short ~ 3 factor
model = smf.ols(formula = 'ESG_MKT ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())




































