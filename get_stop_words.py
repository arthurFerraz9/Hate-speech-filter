

def get_stop_words(name):
    file = open(name,'r')
    words = [word.strip() for word in file]
    return words

