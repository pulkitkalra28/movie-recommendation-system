import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('movies.csv')

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
                    movies_data['cast'] + ' ' + movies_data['director']
# print(combined_features)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)
# print(similarity)

movie_name = input("Enter you favourite movie name: ")
list_of_all_titles = movies_data['title'].tolist()
# print(list_of_all_titles)
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
# print(index_of_the_movie)

similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
# print(sorted_similar_movies)

# print("Movies suggested for you: \n")
i = 0
suggested_movies = []
for _ in range(30):
    sorted_similar_movies_index = sorted_similar_movies[i][0]
    # title_from_index = movies_data[movies_data.index == sorted_similar_movies_index]['title'].values[0]
    suggested_movies.append(movies_data[movies_data.index == sorted_similar_movies_index])
    i += 1
    # print(i, ' ', title_from_index)

suggested_movies_dataset = pd.concat(suggested_movies)
suggested_movies_dataset.to_csv("/Users/mohit/OneDrive/Desktop/suggested_movies.csv", index=False)
