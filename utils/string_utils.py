import nltk

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')  # 特殊符号

stemmer = nltk.stem.PorterStemmer()


def stem(word):
    return stemmer.stem(word)


def clean_sentence(text, stemming=False):
    """
    去除字符串中的特殊字符
    :param text:字符串
    :param stemming:提取词根，将相同单词的不同形式映射到同一单词上
    :return:
    """
    for token in punct:  # 去除特殊字符
        text = text.replace(token, "")
    words = text.split()
    if stemming:  # 对单词进行归一化
        stemmed_words = []
        for w in words:
            stemmed_words.append(stem(w))  # stem(w)将相同意思的单词指向同一单词(提取词根)
        words = stemmed_words
    return " ".join(words)


def clean_name(name):
    """
    # 去除字符串中的".", "-"，用"_"连接
    :param name:
    :return:
    """
    if name is None:
        return ""
    for token in punct:  # 去除姓名中的特殊字符
        name = name.replace(token, "")
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
    return "_".join(x)


def build_pair_features(name, item):
    """
    构建消歧特征对
    :param name:
    :param item:
    :return:
    """
    res = []
    if len(item) > 0:
        for x in item:
            name.append(x)
            res.append("_".join(name))
            name.pop()  # 删除列表末尾元素
    return res



