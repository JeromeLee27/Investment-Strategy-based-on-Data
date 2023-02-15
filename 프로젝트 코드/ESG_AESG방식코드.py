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

#ESG등급 이용하는 부분
#----------------------
#ESG등급 부여된 것들 중 'A+, A, B+, C, D'에 해당되는 것만 가져온다.
combined_want = combined[(combined['ESG등급'] == 'A+') | (combined['ESG등급'] == 'A') | (combined['ESG등급'] == 'B+') | (combined['ESG등급'] == 'C') | (combined['ESG등급'] == 'D')]
combined_rate = combined_want[['코드', 'ESG등급']]

#A+ -> 5, A -> 4, B+ -> 3, C -> 2, D -> 1로 값 변경
combined_rate.replace('A+', 5, inplace = True)
combined_rate.replace('A', 4, inplace = True)
combined_rate.replace('B+', 3, inplace = True)
combined_rate.replace('C', 2, inplace = True)
combined_rate.replace('D', 1, inplace = True)

#시가총액 변환하는 부분
#---------------------
#시총규모 데이터 로드
size_win = pd.read_excel('시총규모.xlsx', sheet_name = 'Long 시총', index_col = 'Date')
size_lose = pd.read_excel('시총규모_re.xlsx', sheet_name = 'Short 시총', index_col = 'Date')
size_esg_win = size_win.copy()
size_esg_lose = size_lose.copy()

size_esg = pd.concat([size_esg_win, size_esg_lose], axis = 1)

#시총 규모를 큰 것부터 순위로 나열
#ascending = True하면서 값이 큰 것을 높은 숫자가 나오게 지정했음. 
Ranking = size_esg.rank(method = 'first', axis = 1, ascending = True)
Weight = Ranking.copy()

numGroup = 5

#시가총액 기준으로 5개의 그룹으로 나눠준다. -> 5그룹이 가장 시총 큰 그룹
for t in tqdm(Time):
    numStock = len(Ranking.loc[t].dropna())
    if numStock % numGroup == 0 :
        for i in range(1, numGroup + 1):            
            boundary1 = numStock*(i-1) / numGroup #하한값
            boundary2 = numStock*i / numGroup #상한값
            Weight.loc[t][(boundary1 < Ranking.loc[t]) & (Ranking.loc[t] <= boundary2)] = i
    else:
        for i in range(1, numGroup + 1):            
            boundary1 = numStock*(i-1) / numGroup #하한값
            boundary2 = numStock*i / numGroup #상한값
            if i == numGroup:
                Weight.loc[t][(boundary1 + 1 < Ranking.loc[t]) & (Ranking.loc[t] <= boundary2)] = i
            else :
                Weight[(boundary1 < Ranking) & (Ranking <= boundary2)] = i

esg_rate = Weight.copy() #시총 순위             

#보기 쉽게 전처리 해주는 과정
esg_rate = esg_rate.T 
esg_rate.reset_index(inplace = True)
esg_rate.rename(columns = {'index' : '코드'}, inplace = True)
esg_rate_com = pd.merge(combined_rate, esg_rate, on = '코드')
esg_rate_com.to_excel('aesg.xlsx')

#AESG = ESG 등급 / (각 종목이 각 시점에 갖는 시가총액 그룹) 을 구하는 과정은
#esg_rate_com.to_excel('aesg.xlsx')를 통해 aesg.xlsx 엑셀 시트에 저장하여 엑셀 내에서 진행해줬다.

#계산된 aesg값을 불러와서 전처리를 해준다.
aesg = pd.read_excel('aesg.xlsx', sheet_name = 'answer', index_col  = '코드')
aesgT = aesg.T
aesgT.drop('ESG등급', axis = 0, inplace = True)
aesgT = aesgT.iloc[2:,]
aesgT.index = pd.to_datetime(aesgT.index)

#aesg점수를 바탕으로 순위를 매기는 과정
aesgT_r = aesgT.rank(method = 'first', axis = 1, ascending = False)
aesgrank = aesgT_r.copy()

#aesg가 최고점 그룹 -> 1 / aesg 최저점 그룹 -> 4로 구분해주는 과정
for t in tqdm(Time):
    numStock = len(aesgrank.loc[t].dropna())
    for i in range(1, numGroup + 1):            
        boundary1 = numStock*(i-1) / numGroup #하한값
        boundary2 = numStock*i / numGroup #상한값
        aesgrank.loc[t][(boundary1 < aesgrank.loc[t]) & (aesgrank.loc[t] <= boundary2)] = i

#기존 분석에 사용된 종목들의 수익률을 하나로 묶어서 로드한다.
r_ = pd.concat([win_r, lose_r], axis = 1)

a_ = aesgrank.copy() ; b_ = r_.copy()
r_comb = r_.copy()

#두 dataframe의 col순서를 맞춰준다. -> 곱셈 계산을 편하게 하게 하려는 목적.
a_ = a_.reindex(sorted(a_.columns), axis=1)
r_comb = r_comb.reindex(sorted(r_comb.columns), axis=1)

aesg_weight = a_.copy()

#CS Momentum Pf 수익률 정보 생성
df_aesg = pd.DataFrame(columns = ['Group' + str(i+1) for i in range(numGroup)], index = Time)
#그룹별 월말 데이터를 만들어주는 것.
for i in range(1, numGroup+1):
    df_Return['Group' +str(i)] = (Return[Group == i] * Weight[Group == i]).sum(axis = 1)
    #전체 Return 중에 그룹의 리턴합을 전체 합쳐 Weight를 곱해준 방식임.
    
#각 그룹에 속하는 종목들의 수익률을 nan값을 제외하고 합치고, 데이터 개수로 나눠준다.    
for t in tqdm(Time):
    df_aesg.loc[t, 'Group1'] = float(np.nansum(r_comb.loc[t,:][aesg_weight.loc[t,:] == 1]) / len(r_comb.loc[t,:][aesg_weight.loc[t,:] == 1]))
    df_aesg.loc[t, 'Group2'] = float(np.nansum(r_comb.loc[t,:][aesg_weight.loc[t,:] == 2]) / len(r_comb.loc[t,:][aesg_weight.loc[t,:] == 2]))
    df_aesg.loc[t, 'Group3'] = float(np.nansum(r_comb.loc[t,:][aesg_weight.loc[t,:] == 3]) / len(r_comb.loc[t,:][aesg_weight.loc[t,:] == 3]))
    df_aesg.loc[t, 'Group4'] = float(np.nansum(r_comb.loc[t,:][aesg_weight.loc[t,:] == 4]) / len(r_comb.loc[t,:][aesg_weight.loc[t,:] == 4]))

#데이터 타입 확인 결과 각 데이터가 object형이었기 때문에 float형으로 변경해줌
df_aesg['Group1'] = df_aesg['Group1'].astype('float') 
df_aesg['Group2'] = df_aesg['Group2'].astype('float') 
df_aesg['Group3'] = df_aesg['Group3'].astype('float') 
df_aesg['Group4'] = df_aesg['Group4'].astype('float') 
   
df_Value = (1+df_aesg/100).cumprod()*100 #초기 투자를 100으로 잡고 누적곱해준 것

df_Maxp = df_Value.cummax()
df_DDp = (df_Value/df_Maxp -1) * 100    

#group1 long, group4 short을 둘다 진행해 구한 수익률, 가격
Rp1 = df_aesg.iloc[:,0] - df_aesg.iloc[:,-1]
Vp1 = (1+Rp1/100).cumprod() *100 ; 
Maxp1 = Vp1.cummax()
DDp1 = (Vp1/Maxp1 - 1)*100

#group1 long만 해보기
Rp2 = df_aesg.iloc[:,0]
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
FactorReg['AESG_MKT'] = FactorReg['Group1'] - FactorReg['MKF2000']

#22년 9월 데이터까지 있기 때문에 이를 맞춰주기 위해서 다음과 같이 slicing진행
FactorReg = FactorReg.iloc[:-9, ]

Rp3 = FactorReg['AESG_MKT']
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

df_aesg['win_lose'] = df_aesg['Group1'] - df_aesg['Group4']

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

#각 전략과 BM performance 비교 그래프
group_show(Vp1, Vp_bm, DDp1, DDp_bm) #win long - lose short
group_show(Vp2, Vp_bm, DDp2, DDp_bm) #win long
group_show(Vp3, Vp_bm, DDp3, DDp_bm) #win long - mkf2000 short

# 3 - Factor Regression OLS
#win-lose ~ 3 factor
model = smf.ols(formula = 'win_lose ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())


#win ~ 3 factor
model = smf.ols(formula = 'Group1 ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())

#win-lose long - mkf2000 short ~ 3 factor
model = smf.ols(formula = 'AESG_MKT ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())



 