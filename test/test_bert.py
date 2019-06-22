def test_download_tokenizer():
    import fenwicks as fw
    tokenizer = fw.bert.get_tokenizer()


def test_tokenize_uncased():
    import fenwicks as fw
    tokenizer = fw.bert.get_tokenizer()
    test_sentence = "This here's an example of using the BERT tokenizer"
    expected_result = ['this', 'here', "'", 's', 'an', 'example', 'of', 'using', 'the', 'bert', 'token', '##izer']
    result = tokenizer.tokenize(test_sentence)
    assert expected_result == result


def test_tokenize_cased():
    import fenwicks as fw
    tokenizer = fw.bert.get_tokenizer('cased_L-12_H-768_A-12')
    test_sentence = "This here's an example of using the BERT tokenizer"
    expected_result = ['This', 'here', "'", 's', 'an', 'example', 'of', 'using', 'the', 'B', '##ER', '##T', 'token',
                       '##izer']
    result = tokenizer.tokenize(test_sentence)
    assert expected_result == result
