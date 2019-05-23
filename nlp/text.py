from ..imports import *

import csv
import nltk
from bs4 import BeautifulSoup

__all__ = ['tsv_lines']


def html_to_words(raw_text):
    txt = BeautifulSoup(raw_text, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", txt)
    words = letters_only.lower().split()
    stops = set(nltk.corpus.stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]
    return " ".join(meaningful_words)


def tsv_lines(input_file, quotechar=None):
    with gfile.GFile(input_file) as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def to_unicode(text):
    return text if isinstance(text, str) else text.decode("utf-8", "ignore")