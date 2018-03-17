def gf(word, blob):
    count = 0
    for doc in blob:
        count += doc.count(word)
    return count