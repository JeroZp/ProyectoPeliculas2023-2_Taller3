from django.shortcuts import render
from movie.models import Movie

from dotenv import load_dotenv
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import os

_ = load_dotenv('../openAI.env')    
openai.api_key  = os.environ['openAI_api_key']

def recommendation(request):
    rec = request.GET.get('req')
    if rec:
        emb = get_embedding(rec, engine='text-embedding-ada-002')
        movies = list(Movie.objects.all())
        sim = []
        for movie in movies:
            emb_binary = np.array(movie.emb).tobytes()
            rec_emb = list(np.frombuffer(emb_binary))
            sim.append(cosine_similarity(emb, rec_emb))
        sim = np.array(sim)
        idx = np.argmax(sim)
        
        list_point = []

        for i in range(len(sim)):
            if sim[i] > 0.8:
                list_point.append({'movie':movies[i] , 'puntaje':sim[i]})

        list_pointS = sorted(list_point, key=lambda x: x["puntaje"], reverse=True)

        list_ordenada_movies = []
        for i in list_pointS:
            list_ordenada_movies.append(i['movie'])

        return render(request, 'recommendation.html', {'movies': list_ordenada_movies})
    
    else:
        return render(request, 'recommendation.html')