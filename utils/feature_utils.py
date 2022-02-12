from itertools import chain

from utils import string_utils


def transform_feature(data, f_name):
    """
    特征切分
    :param data:输入特征
    :param f_name: 特征属性
    :return: __f_name__特征单词
    """
    if type(data) is str:
        data = data.split()
    assert type(data) is list  # data是一个str列表
    features = []
    for d in data:
        features.append("__%s__%s" % (f_name.upper(), d))  # *
    return features


def extract_common_features(item):
    """
    提取公共特征
    :param item:
    :return:
    """
    title_features = []
    title_features.extend(string_utils.clean_sentence(item["title"], stemming=True).lower().split())
    # keywords_features = []
    # keywords = item.get("keywords")
    # if keywords:
    #     keywords_features.extend([string_utils.clean_name(k) for k in keywords])  # 去除字符串中的".", "-"，用"_"连接
    venue_features = []
    venue_name = item.get('venue', '')
    if len(venue_name) > 2:
        venue_features.append(string_utils.clean_sentence(venue_name.lower()))
    year_features = []
    year = item.get('year')
    if year:
        year_features.append(year)
    # abs_features = []
    # abs = item.get("abstract")
    # if abs:
    #     abs_features.extend(string_utils.clean_sentence(abs.lower()))
    return title_features, venue_features, year_features


def extract_author_features(item, order=None):
    """
    提取作者属性特征
    :param item:
    :param order:
    :return:
    """
    # 提取公共特征:标题、关键字、出版社、年份、摘要
    title_features, venue_features, year_features = extract_common_features(item)
    author_features = []
    org_coauthor_features = []
    org_title_features = []
    org_venue_features = []
    org_year_features = []
    for i, author in enumerate(item["authors"]):  # 枚举第i个作者
        if order is not None and i != order:  # 找到所要的第order个作者 *
            continue
        name_feature = []
        org_feature = []
        coauthor_feature = []
        name = author.get("name", "")
        org_name = author.get("org", "")
        if len(name) > 2:
            name_feature.extend(
                transform_feature([string_utils.clean_name(name)], "name")
            )
        if len(org_name) > 2:
            org_feature.append(
                string_utils.clean_name(org_name)  # 返回nankai_univ形式
            )

        for j, coauthor in enumerate(item["authors"]):
            if i == j:
                continue
            coauthor_name = coauthor.get("name", "")  # 获得合作者名字
            if len(coauthor_name) > 2:
                coauthor_feature.append(
                    string_utils.clean_name(coauthor_name)
                )
        # 构建消歧特征对
        org_coauthor_features.extend(transform_feature([k for k in string_utils.build_pair_features(org_feature, coauthor_feature)], "org_coauthor"))  # org_coauthor_features
        org_title_features.extend(transform_feature([k for k in string_utils.build_pair_features(org_feature, title_features)], "org_title"))  # org_title_features
        org_venue_features.extend(transform_feature([k for k in string_utils.build_pair_features(org_feature, venue_features)], "org_venue"))  # org_venue_features
        org_year_features.extend(transform_feature([k for k in string_utils.build_pair_features(org_feature, year_features)], "org_year"))  # org_year_features
        # org_keywords_features.extend(transform_feature([k for k in string_utils.build_pair_features(org_feature, keywords_features)], "org_keyword"))  # org_keywords_features
        # 将消歧特征对加入作者特征中
        author_features.append(
            name_feature + org_coauthor_features + org_title_features + org_venue_features + org_year_features)
    author_features = list(chain.from_iterable(author_features))
    return author_features
