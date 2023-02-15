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

#price와 ESG등급코드를 맞춰주기
price = price.T
price = price.reset_index()  
price.rename(columns = {'index' : '코드'}, inplace = True)

#계산 편하게 하기 위해 datetime변경
win.index = pd.to_datetime(win.index)
lose.index = pd.to_datetime(lose.index)

#비교군인 BM와 RF데이터도 정리
rawBM.index = pd.to_datetime(rawBM.index)
rawRF.index = pd.to_datetime(rawRF.index)


BM = rawBM.resample('BM').last().copy()
MonthRF = rawRF.resample('BM').last().copy()

#월별 수익률 확인하는 작업
RF = MonthRF.shift(1)/12 
bmReturn = (BM.diff() / BM.shift(1))*100

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

#-------------------------------------

#KRX데이터 보여주는 pykrx 라이브러리 로드
from pykrx import stock

#2022년 11월 27일에 상장되어 있는 전체 주식 데이터를 로드
codes = stock.get_market_ticker_list('221127', market = 'ALL')

#각 종목의 코드와 종목명을 corp변수에 append해준다.
corp = []
for code in codes :    
    name = stock.get_market_ticker_name(code)
    corp.append([code, name])
    
corp = pd.DataFrame(corp, columns = ['코드명', '회사명'])
#기존 데이터와 맞춰주기 위해 앞에 'A'더해준다.
corp['코드명'] = 'A' + corp['코드명']

#ESG보고서 제작한 종목들을 로드 / <KRX ESG포털>에서 자체 드레그해서 크롤링했음. 
esg_paper = pd.read_excel('ESG보고서_제작.xlsx') ; esg_paper

#3개년 데이터를 썼기 때문에 중복 종목들은 drop해준다.
esg_paper.drop_duplicates('회사명', keep = 'first', inplace =True)

#esg_paper와 corp 데이터를 회사명을 기준으로 합쳐준다
esg_paper_new = pd.merge(esg_paper, corp, on = '회사명')

#ESG등급표가 존재하는 전체 종목들 중 기존에 이용했던 등급들의 종목인 win과 lose 데이터를 esg_pp로 합쳐준다.
esg_pp = pd.concat([win, lose], axis = 0)
esg_paper_new.rename(columns = {'코드명' : '코드'}, inplace =True)
#기존 분석에서 이용된 ESG등급 존재 종목들과 esg_paper를 합쳐준다.
esg_new = pd.merge(esg_paper_new, esg_pp, on = '코드', how = 'outer')

#기존 데이터들 중 esg_paper발간 종목과 비발간 종목들을 나눠서 새롭게 받아준다.
esg_paper_ok = esg_new[(~esg_new['회사명'].isnull()) & (~esg_new['기업명'].isnull())] #전체 8개의 종목 제외하곤 113개의 종목이 a+,a,b+,c,d의 종목에 포함이 돼.
esg_paper_no = esg_new[esg_new['회사명'].isnull()]

#각 종목군과 각 종목의 가격을 합쳐준다.
com_ok = pd.merge(esg_paper_ok, price,  how = 'inner', on = '코드')
com_no = pd.merge(esg_paper_no, price,  how = 'inner', on = '코드')

#필요없는 열들 drop
com_ok.drop(['Unnamed: 0', '기업명', 'ESG등급', '환경', '사회', '지배구조', '회사명', '발행년도',
       '업종','제 3자 검증'], axis = 1, inplace = True)
com_no.drop(['Unnamed: 0', '기업명', 'ESG등급', '환경', '사회', '지배구조', '회사명', '발행년도',
       '업종','제 3자 검증'], axis = 1, inplace = True)

#dataframe을 전치해주고, 열 지정, 필요없는 변수들을 drop 해준다
win = com_ok.T ; lose = com_no.T
win.columns = win.iloc[0, :] 
lose.columns = lose.iloc[0, :]
win.drop('코드', axis = 0, inplace = True)
lose.drop('코드', axis = 0, inplace = True)


#계산 편하게 하기 위해 datetime변경
win.index = pd.to_datetime(win.index)
lose.index = pd.to_datetime(lose.index)

win = win.resample('BM').last().copy()
lose = lose.resample('BM').last().copy()

win_r = (win.diff() / win.shift(1)) * 100
lose_r = (lose.diff() / lose.shift(1)) * 100

stDateNum = 20100101
stDate = pd.to_datetime(str(stDateNum), format = '%Y%m%d')

win_r = win_r[win_r.index >= stDate].copy()
lose_r = lose_r[lose_r.index >= stDate].copy()

win_group = win_r.copy()
win_weight = win_r.copy()
lose_group = lose_r.copy()
lose_weight = lose_r.copy()

Time = win_r.index 

Time = Time[Time>= stDate].copy()
bmReturn = bmReturn[bmReturn.index >= stDate].copy()
RF = RF[RF.index >= stDate].copy()

#해당 시기에 없던 데이터는 제외하고, 있는 값들만 뽑아서 해당 데이터들로 포트폴리오 짤 때 1/n개화 해주는 작업.
for t in tqdm(Time):
    numStock1 = len(win_group.loc[t].dropna())
    if numStock1 != 0 :
        win_weight.loc[t][~win_group.loc[t].isnull()] = 1/numStock1   # nan
    
for t in tqdm(Time):
    numStock2 = len(lose_group.loc[t].dropna())
    if numStock2 != 0 :
        lose_weight.loc[t][~lose_group.loc[t].isnull()] = 1/numStock2   # nan
        
win_weight = win_weight.shift(1) ; lose_weight = lose_weight.shift(1)

#paper 존재 o, 존재 x 포지션별로 수익률 데이터 합해주기
df_Return = pd.DataFrame(columns = ['ESG_paper_ok', 'ESG_paper_no'], index = Time)
#각 종목별 sum(해당 월 수익률 * 해당 월 전체 중 비율)해서 월별 데이터로 뽑아내기 과정 
win_accum = (win_group * win_weight) ; lose_accum = (lose_group * lose_weight)

df_Return['ESG_paper_ok'] = win_accum.sum(axis = 1)
df_Return['ESG_paper_no'] = lose_accum.sum(axis = 1)
df_Return['ok_no'] = df_Return['ESG_paper_ok'] - df_Return['ESG_paper_no']
df_Value = (1+df_Return/100).cumprod()*100 #초기 투자를 100으로 잡고 누적곱해준 것

df_Maxp = df_Value.cummax()
df_DDp = (df_Value/df_Maxp -1) * 100    

#long-short 둘다
Rp1 = df_Return['ok_no']
Vp1 = (1+Rp1/100).cumprod() *100 ; 
Maxp1 = Vp1.cummax()
DDp1 = (Vp1/Maxp1 - 1)*100

#long만 해보기
Rp2 = df_Return.iloc[:,0]
Vp2 = (1+Rp2/100).cumprod() *100 ; 
Maxp2 = Vp2.cummax()
DDp2 = (Vp2/Maxp2 - 1)*100

#Benchmark 수익률
Rp_bm = bmReturn.copy()
Rp_bm = Rp_bm - RF
Vp_bm = (1+Rp_bm.shift(-1)/100).cumprod() * 100
Vp_bm = Vp_bm.shift(1) ; Vp_bm[0] = 100 
Maxp_bm = Vp_bm.cummax()
DDp_bm = (Vp_bm / Maxp_bm -1) * 100

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
FactorReg['ok_MKT'] = FactorReg['ESG_paper_ok'] - FactorReg['MKF2000']

#22년 9월 데이터까지 있기 때문에 이를 맞춰주기 위해서 다음과 같이 slicing진행
FactorReg = FactorReg.iloc[:-9, ]

#win long, MKF2000 short을 진행해 구한 수익률, 가격 
#바로 위에서 FactorReg dataframe의 'ESG_MKT'변수
#FactorReg['ESG_MKT'] = FactorReg['win_lose'] - FactorReg['MKF2000']
#에서 이용한 걸 그대로 사용

#ESG paper ok - MKF2000
Rp3 = FactorReg['ok_MKT']
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

#각 전략과 BM performance 비교 그래프
group_show(Vp1, Vp_bm, DDp1, DDp_bm) #win long - lose short
group_show(Vp2, Vp_bm, DDp2, DDp_bm) #win long
group_show(Vp3, Vp_bm, DDp3, DDp_bm) #win long - mkf2000 short

#--------------------------------------------------------------------------------
#ESG팩터에 대한 팩터회귀분석과 그 결과에 대한 해석
#--------------------------------------------------------------------------------
# 3 - Factor Regression OLS
#win-lose ~ 3 factor
model = smf.ols(formula = 'ok_no ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())


#win ~ 3 factor
model = smf.ols(formula = 'ESG_paper_ok ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())

#win-lose long - mkf2000 short ~ 3 factor
model = smf.ols(formula = 'ok_MKT ~ MKT + SMB + HML', data = FactorReg)
result = model.fit()
print(result.summary())