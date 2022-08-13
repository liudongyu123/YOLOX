# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 20、递归函数实现斐波那契数列.py
@time: 2022/2/15 19:35
"""


def fib(n):
    if n <= 2:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


print(fib(10))
