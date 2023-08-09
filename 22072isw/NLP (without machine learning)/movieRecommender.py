import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = pd.read_csv('movies_metadata.csv', low_memory=False)
data['overview'] = data['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title_to_index = dict(zip(data['title'].str.lower(), data.index))

def get_recommendations(title, cosine_sim=cosine_sim) :
    
    title = title.lower()
    
    if title not in title_to_index :
        print("no such movie in the list!")
        return
    
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x : x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    print(sim_scores)
    movie_indices = [idx[0] for idx in sim_scores]
    print(data['title'].iloc[movie_indices])
    
while True :
    get_recommendations(input(">> "))