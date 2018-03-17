from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize


# Takes a file handler and returns a list of cleaned up sentences
def cleanup(fh=None):
    if fh is not None:
        fcontent = fh.read()
        lines = sent_tokenize(fcontent)
        tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
        stops = set(stopwords.words('english'))
        punctuations = ['.', ',', '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%',
                        '^', '&', '*', '(', ')', '-', '_', '=', '+', '—']
        stops.update(punctuations)
        cleaned_sentences = []
        for l in lines:
            tokenized_line = tokenizer.tokenize(l)
            cleaned_line = [word for word in tokenized_line if word not in stops]
            cleaned_sentences.append(cleaned_line)
        cleaned_sentences = [line for line in cleaned_sentences if line != []]
        return cleaned_sentences
    else:
        return None


# Takes a list of cleaned up documents and  consolidates them into a list of strings, each string being a document
def fusion(blob):
    textblob = []
    for doc in blob:
        text = ""
        for line in doc:
            text = text + " ".join(line) + " "
        textblob.append(text)
    return textblob


# Takes a list of cleaned up documents and consolidates them into a list of Bag of Words.
def wordify(blob):
    bow = []
    for doc in blob:
        words = []
        for line in doc:
            for word in line:
                words.append(word)
        bow.append(words)
    return bow


# Takes a list of Bags of Words and combines them into one
def baggify(blob):
    bow = []
    for doc in blob:
        bow += list(set(doc))
    return set(bow)