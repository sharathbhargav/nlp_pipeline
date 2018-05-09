import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from collections import Counter

removableWords = set(stopwords.words('english'))

extraWords = ['.', ',', '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^',
                '&', '*', '(', ')', '-', '_', '=', '+', '—', ' ','Reddit','reddit','Lol','Nah','I']

removableWords.update(extraWords)


nlp = spacy.load('/home/sharathbhragav/anaconda3/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')

testDocument=open("/media/sharathbhragav/New Volume/redditPosts/hot/18-04-21_0","r")
data=testDocument.read()
data=data.replace("\n"," ")
doc=nlp(data)
count1=0
count2=0
out=""
noun=[]
for each in doc.noun_chunks:
    count1+=1
    out=each
    if str(each).lower() not in removableWords:
        #print(each)
        noun.append(str(each))
        count2+=1
nounSet=set(noun)

freq=Counter(noun)
print(freq.most_common(25))
print(len(noun))
print(len(nounSet))

print("count1",count1)

print("count2",count2)
#print(doc)
#print(data)