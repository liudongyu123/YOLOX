# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 23、递归阶乘求和.py
@time: 2022/2/15 19:54
"""


def factor(n):
    if n <= 1:
        return 1
    else:
        return n * factor(n - 1)


sum_num = 0
for i in range(1, 11):
    sum_num += factor(i)
print(sum_num)
