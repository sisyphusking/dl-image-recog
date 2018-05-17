
def get_english_words():
    word_list = []
    with open('./data/common_english.txt', 'r') as f:
        for word in f.readlines():
            word = word.strip('\n')
            if len(word) > 4 and len(word) <= 9 and "-" not in word:
                word_list.append(word.lower())
    return word_list



