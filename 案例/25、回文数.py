# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 25、回文数.py
@time: 2022/2/16 9:10
"""


def is_palindrome(x):
    if x < 0 or x > 0 and x % 10 == 0:
        return False
    str_x = str(x)
    return str_x == str_x[::-1]


print(is_palindrome(1221))
