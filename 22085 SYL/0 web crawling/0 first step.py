import requests
from bs4 import BeautifulSoup as bs
import sys
import pickle
'''
page = requests.get('https://music.bugs.co.kr/chart')
soup = bs(page.text, 'html.parser')

with open('./100chartHTML.html', 'wb') as f:
    pickle.dump(soup, f)
'''
with open('./100chartHTML.html', 'rb') as f:
    soup = pickle.load(f)

elements = soup.find_all(attrs={'class': 'title', 'adult_yn': 'N'})

ids = []
for i in elements:
    st = str(i)
    idx = st.index("bugs.music.listen('")+19
    end = st.index("',true);")
    ids.append(st[idx:end])
    print(ids[-1])
    # song ids

lyricPage = requests.get(f'https://music.bugs.co.kr/track/{ids[0]}')
lyricSoup = bs(lyricPage.text, 'html.parser')

print(str(lyricSoup.select('xmp'))[6:-7])
