import re
import nltk
from bs4 import BeautifulSoup


def text_to_words(raw_text):
    txt = BeautifulSoup(raw_text, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", txt)
    words = letters_only.lower().split()
    stops = set(nltk.corpus.stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)
