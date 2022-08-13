# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 19、深入理解递归函数.py
@time: 2022/2/15 19:28
"""


def p(n):
    if n == 0:
        return
    print("递归前：--->", n)
    p(n - 1)
    print("递归后：--->", n)


p(5)
