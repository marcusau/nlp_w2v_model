from typing import List,Dict,Union,Tuple
import multiprocessing
import re,json
import numpy as np

import datetime
import time

from sklearn.metrics.pairwise import cosine_similarity
from pymagnitudelight import Magnitude, FeaturizerMagnitude, MagnitudeUtils

pymaganitute_model_light_filepath=r'C:\Users\marcus\PycharmProjects\word2vec_training\model\magnitude\boc_app_light.magnitude'


model = Magnitude(pymaganitute_model_light_filepath, eager=True, case_insensitive=False,  batch_size=500000, supress_warnings=True, normalized=True,  language=None)

word='林鄭'
topn=10
topn_words=model.most_similar(positive=word,topn=topn,)

for w in topn_words:
    print(w)