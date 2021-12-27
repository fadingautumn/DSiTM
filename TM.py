import pymorphy2
import re
import gensim
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore, LsiModel
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
morph = pymorphy2.MorphAnalyzer()

text = open("C:\\Users\\zen15\\Desktop\\DSiTM\\full.txt", encoding = 'utf8').read()

# Подготовка текста
cleantext = text.lower()
cleantext = re.sub('[^а-яА-Я]', ' ', cleantext)
cleantext = re.sub(r'\s+', ' ', cleantext)
justokens = word_tokenize(cleantext)

filtlm = []
SW = stopwords.words('russian')
for w in justokens:
    if w not in SW:
        filtlm.append(w)

lemmtok = []
for word in filtlm:
    temp1 = []
    pas = morph.parse(word)[0]
    temp1.append(pas.normal_form)
    lemmtok.append(temp1)

wow = []
for wr in lemmtok:
    for lm in wr:
        wow.append(lm)
        
#Биграммы и тритграммы

finder = BigramCollocationFinder.from_words(wow)
finder.apply_freq_filter(3)
bigr = finder.nbest(bigram_measures.pmi, 1500)
print (finder.nbest(bigram_measures.pmi, 10))

finder2 = TrigramCollocationFinder.from_words(wow)
finder2.apply_freq_filter(3)
trigr = finder2.nbest(trigram_measures.pmi, 200)
print (finder2.nbest(trigram_measures.pmi, 10))

#LDA модель
dictionary1 = corpora.Dictionary (lemmtok)
corpus1 = [dictionary.doc2bow(l) for l in lemmtok]
dictionary = corpora.Dictionary (bigr)
corpus = [dictionary.doc2bow(l) for l in bigr]

LDA_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
  
#Показать темы
print(LDA_model.print_topics(-1))

#LSA модель

LSA_model = LsiModel(corpus, num_topics=10, id2word=dictionary)
topics = LSA_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
    
#NMF модель
from gensim.models import Nmf
nmf = Nmf(corpus1, num_topics = 10)
for topic in nmf.print_topics():
    print(topic)
    
#U_Mass для LDA, LSA и NMF
coherence_model_lda = CoherenceModel(model=LDA_model, texts=lemmtok, dictionary=dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score LDA: ', coherence_lda)

coherence_model_lsa = CoherenceModel(model=LSA_model, texts=lemmtok, dictionary=dictionary, coherence="u_mass")
coherence_lsa = coherence_model_lsa.get_coherence()
print('\nCoherence Score LSA: ', coherence_lsa)

coherence_model_nmf = CoherenceModel(model=nmf, texts=lemmtok, dictionary=dictionary1, coherence="u_mass")
coherence_nmf = coherence_model_nmf.get_coherence()
print('\nCoherence Score NMF: ', coherence_nmf)


#Перплексия и когерентность
from gensim.models.coherencemodel import CoherenceModel

coherence_model_lda = CoherenceModel(model=LDA_model, texts=lemmtok, dictionary=dictionary, coherence="c_v")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score LDA: ', coherence_lda)
print('\nPerplexity Score LDA: ', LDA_model.log_perplexity(corpus))

coherence_model_lsa = CoherenceModel(model=LSA_model, texts=lemmtok, dictionary=dictionary, coherence="c_v")
coherence_lsa = coherence_model_lsa.get_coherence()
print('\nCoherence Score LSA: ', coherence_lsa)

coherence_model_nmf = CoherenceModel(model=nmf, texts=lemmtok, dictionary=dictionary, coherence="c_v")
coherence_nmf = coherence_model_nmf.get_coherence()
print('\nCoherence Score NMF: ', coherence_nmf)
