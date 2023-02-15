#라이브러리 로드
import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}

#ESG 등급 평가 데이터 크롤링 위해 url 나눠서 지정. url1 뒤의 pg만 바뀌기 때문에 for문으로 통해 page별 data crawling 예정
url1 = 'http://www.cgs.or.kr/business/esg_tab04.jsp?pg='
url2 = '&pp=10&skey=&svalue=&sfyear=2021&styear=2021&sgtype=&sgrade=#ui_contents'

#빈 list data를 만들어주고, 크롤링 통해 나온 데이터 data에 삽입시켜준다.
data = [] 
for i in range(1, 102) :
    url = url1 + str(i) + url2
    req = requests.get(url, headers = headers)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    for i in range(1,11):
        data.append(soup.select('tr')[i].text.split('\n')[2:8])
        
#크롤링한 데이터를 dataframe으로 만들어 주고 데이터 확인
data = pd.DataFrame(data, columns = ['기업명', '코드', 'ESG등급', '환경', '사회', '지배구조'])
data.head()

#이를 엑셀로 저장해 분석에 이용
data.to_excel('ESG기업등급표.xlsx')