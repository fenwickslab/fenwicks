import re
import csv
import nltk
from bs4 import BeautifulSoup
import tensorflow as tf

__all__ = ['tsv_lines']


def html_to_words(raw_text):
    txt = BeautifulSoup(raw_text, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", txt)
    words = letters_only.lower().split()
    stops = set(nltk.corpus.stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]
    return " ".join(meaningful_words)


def tsv_lines(input_file, quotechar=None):
    with tf.io.gfile.GFile(input_file) as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines
