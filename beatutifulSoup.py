import bs4 as bs
import urllib.request
import re


def printParagraphs(link,filenameToSave):
    file= open(filenameToSave,"w")
    source = urllib.request.urlopen(link).read()
    htmlCode = bs.BeautifulSoup(source, 'lxml')
    for par in htmlCode.find_all('p'):
        print(par.text)
        file.write(par.text)
    file.write("\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    file.close()
    for links in htmlCode.find_all('a', attrs={'href': re.compile('https://')}):
        print(links.get('href'))
        #printParagraphs(links.get('href'))


inputWord = input("Enter word to search")

printParagraphs('https://en.wikipedia.org/wiki/'+inputWord,inputWord)


#source=urllib.request.urlopen('https://en.wikipedia.org/wiki/Word2vec').read()
#tmlCode=bs.BeautifulSoup(source,'lxml')
#print(htmlCode)

#paragraph extraction
#for par in htmlCode.find_all('p'):
#   #print(par.text)

#or links in htmlCode.find_all('a',attrs={'href':re.compile('https://')}):
#
#   print(links.get('href'))
