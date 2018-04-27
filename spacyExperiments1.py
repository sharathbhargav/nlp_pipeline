import spacy
from spacy import displacy
from collections import Counter
import clusteringExperiments2 as exp2
nlp = spacy.load('/home/sharathbhragav/anaconda3/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')
document1=open("datasets/custom2/Sci/1","r")
text1=document1.read()
doc=nlp(text1)
#print(doc)
#displacy.serve(doc, style='ent')

pathForData="datasets/custom2/"

(clusterCount,labels)=exp2.customKMeansComplete()

fileNameDictionary=exp2.getDocClustersNames(clusterCount,labels)
print(fileNameDictionary)

organizations={}
persons={}
places={}
nouns={}

print(">>>>>>>>>>>>>>>>>",fileNameDictionary[2][0][0:-1])

for i in range(clusterCount):
    clusterOrganization=[]
    clusterPerson=[]
    clusterPlace=[]
    clusterNoun=[]
    for docName in fileNameDictionary[i]:
        tempStr=docName.split('-')
        docTemp=open(pathForData+tempStr[0]+"/"+tempStr[1],"r")
        doc=nlp(docTemp.read())
        clusterNoun.append(list(doc.noun_chunks))
        for ent in doc.ents:
            #print(ent.text, ent.start_char, ent.end_char, ent.label_)
            if ent.label_ == "ORG":
                clusterOrganization.append(ent.text)
            if ent.label_ == 'PERSON':
                clusterPerson.append(ent.text)
            if ent.label_ == "GPE":
                clusterPlace.append(ent.text)

    organizations[i]=clusterOrganization
    persons[i]=clusterPlace
    places[i]=clusterPlace
    nouns[i]=clusterNoun

#nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]

# five most common tokens

for i1 in range(clusterCount):

    print(">>>>>>>>>>>>>>>>>>>>>>>Details of cluster ",i1)
    organizations_freq = Counter(organizations[i1])
    #common_words = organizations_freq.most_common(5)
    persons_freq=Counter(persons[i1])
    places_freq=Counter(places[i1])

    print(organizations_freq.most_common(5))
    print(persons_freq.most_common(5))
    print(places_freq.most_common(5))
    print("Nouns====",nouns[i])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")