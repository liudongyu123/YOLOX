# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 17、单词分析.py
@time: 2022/2/3 16:25
"""


def analyse_words(words):
    word_dict = {}
    for i in words:
        if i in word_dict:
            word_dict[i] += 1
        else:
            word_dict[i] = 1
    # print(word_dict)
    max_key = max(word_dict, key=word_dict.get)  # 获取字典中value最大时对应的key值
    print(word_dict)
    print(max_key)
    print(word_dict[max_key])


analyse_words("helloworld")
