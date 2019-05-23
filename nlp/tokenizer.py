import collections
import unicodedata
from typing import List, Dict, Tuple

import tensorflow as tf

from .. import core
from . import text

TextFeat = collections.namedtuple('TextFeat', ['input_ids', 'input_mask'])


def load_vocab(vocab_fn: str) -> Dict:
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_fn) as reader:
        while True:
            token = text.to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    return text.split() if text else []


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        if unicodedata.category(char) != "Mn":
            output.append(char)
    return "".join(output)


def is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    return cat == "Zs"


def is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    return cat in ("Cc", "Cf")


def is_punctuation(char):
    cp = ord(char)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    cat = unicodedata.category(char)
    return cat.startswith("P")


def split_on_punc(text):
    chars = list(text)
    start_new_word = True
    output = []

    for i, char in enumerate(chars):
        if is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)

    return ["".join(x) for x in output]


def clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        output.append(" " if is_whitespace(char) else char)
    return "".join(output)


class BasicTokenizer:
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = text.to_unicode(text)
        text = clean_text(text)
        orig_tokens = whitespace_tokenize(text)

        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = strip_accents(token)
            split_tokens.extend(split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))  # todo:???
        return output_tokens


class WordpieceTokenizer:
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        text = text.to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertTokenizer:
    def __init__(self, vocab_fn: str, do_lower_case: bool = True):
        self.vocab = load_vocab(vocab_fn)
        self.inv_vocab = core.inverse_dict(self.vocab)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text: str) -> List[str]:
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return core.convert_by_dict(self.vocab, tokens)

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return core.convert_by_dict(self.inv_vocab, ids)

    def process_sentence(self, text: str, max_seq_len: int) -> Tuple[List[int], List[int]]:
        tokens_a = self.tokenize(text)

        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[0: max_seq_len - 2]

        tokens = ["[CLS]"]
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids = self.tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        if len(input_ids) < max_seq_len:
            pad_len = max_seq_len - len(input_ids)
            input_ids.extend([0] * pad_len)
            input_mask.extend([0] * pad_len)

        return input_ids, input_mask
