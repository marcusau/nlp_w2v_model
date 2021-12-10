import gensim

from gensim.models import KeyedVectors

gensim_model_path=r'C:\Users\marcus\PycharmProjects\word2vec_training\model\gensim\etnet_w2v.bin'

model = KeyedVectors.load_word2vec_format(gensim_model_path, binary=True)

word='林鄭'
topn=10
topn_words=model.most_similar(word,topn=topn)

for w in topn_words:
    print(w)
