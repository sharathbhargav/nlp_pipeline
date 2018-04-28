import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('/home/sharathbhragav/anaconda3/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')

testDocument=open("documents/news1Hindu","r")
data=testDocument.read()
data=data.replace("\n"," ")
doc=nlp(data)
for token in doc.ents:

    print(token,">>>>>>>>>>")
#print(doc)
print(data)