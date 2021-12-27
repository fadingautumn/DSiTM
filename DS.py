# Импорт
import nltk
import re
import gensim
import numpy as np
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords

# Необработанный текст
text = open("C:\\Users\\zen15\\Desktop\\DSiTM\\full.txt", encoding = 'utf8').read()

# Очистка текста
cleantext = text.lower()
cleantext = re.sub('[^а-яА-Я]', ' ', cleantext)
cleantext = re.sub(r'\s+', ' ', cleantext)

# Токенизация
all_sentences = nltk.sent_tokenize(cleantext)
all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Удаление стоп-слов
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('russian')]
    
# Обучение модели Word2Vec с CBOW и SkipGram

from gensim.models import Word2Vec
w2v_cbow = Word2Vec(min_count=2, window=2, sg = 0)
w2v_cbow.build_vocab(all_words)
w2v_cbow.train(all_words, total_examples=w2v_cbow.corpus_count, epochs= 50)
w2v_sg = Word2Vec(min_count=2, window=2, sg = 1)
w2v_sg.build_vocab(all_words)
w2v_sg.train(all_words, total_examples=w2v_sg.corpus_count, epochs= 50)

#Косинусоидное сходство
print ("Косинусное сходство CBOW пример 1:",w2v_cbow.wv.most_similar('делать', 'быстрее'), '\n')
print ("Косинусное сходство CBOW пример 2:",w2v_cbow.wv.most_similar('завтра', 'солнце'), '\n')
print ("Косинусное сходство SkipGram пример 1:",w2v_sg.wv.most_similar('делать', 'быстрее'), '\n')
print ("Косинусное сходство SkipGram пример 2:",w2v_sg.wv.most_similar('завтра', 'солнце'), '\n')
#Евклидово расстояние
print ("Евклидово расстояние CBOW пример 1:", np.linalg.norm(w2v_cbow.wv['делать'] - w2v_cbow.wv['быстрее']), '\n')
print ("Евклидово расстояние CBOW пример 2:", np.linalg.norm(w2v_cbow.wv['завтра'] - w2v_cbow.wv['солнце']), '\n')
print ("Евклидово расстояние SkipGram пример 1:", np.linalg.norm(w2v_sg.wv['делать'] - w2v_sg.wv['быстрее']), '\n')
print ("Евклидово расстояние SkipGram пример 2:", np.linalg.norm(w2v_sg.wv['завтра'] - w2v_sg.wv['солнце']), '\n')


from nltk.text import Text
text = open("C:\\Users\\zen15\\Desktop\\DSiTM\\full.txt", encoding = 'utf8').read()

# Очистка текста (здесь почему-то надо очищать иначе, ещё не разобрался)
cleantext = re.sub('[^а-яА-Я]', ' ', text)
cleantext = re.sub(r'\s+', ' ', cleantext)
def tokenize(sentences):
    for sent in nltk.sent_tokenize(sentences.lower()):
        for word in nltk.word_tokenize(sent):
            if word not in stopwords.words('russian'):
                yield word

text1 = nltk.Text(tkn for tkn in tokenize(cleantext))
text1.collocations(num=50)

from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

postokens = pos_tag(word_tokenize(cleantext), lang='rus')

#Биграммы и триграммы

finder = BigramCollocationFinder.from_words(postokens)
finder.apply_freq_filter(3)
print (finder.nbest(bigram_measures.pmi, 10))

finder2 = TrigramCollocationFinder.from_words(postokens)
finder2.apply_freq_filter(3)
print (finder2.nbest(trigram_measures.pmi, 10))


#Частеречный фильтр для коллокатов
for word in postokens:
    sym = word[0]
    tag = word[1]
    if tag.find('A') != -1:
        word1 = sym
    if tag.find('S') != -1:
            word2 = sym
            if word1 != 0:
                print (word1, '', word2)
                word1 = 0       
                word2 = 0
                
#Кандидаты в метки тем

print (w2v_cbow.wv.most_similar ('друг'), '\n')
print (w2v_cbow.wv.most_similar ('любовь'), '\n')
print (w2v_cbow.wv.most_similar ('год'), '\n')
print (w2v_cbow.wv.most_similar ('любить'), '\n')
print (w2v_cbow.wv.most_similar ('сердце'), '\n')
