# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 22、因子之和“完数”.py
@time: 2022/2/15 19:50
"""


def factor_num(n):
    s_sum = 0
    for i in range(1, n):
        if n % i == 0:
            s_sum += i
    return s_sum


for j in range(1, 1000):
    if j == factor_num(j):
        print("完数：", j)
