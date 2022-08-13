# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 16、大衍数列.py
@time: 2022/2/3 16:14
"""
for i in range(1, 101):
    num = 0
    if i % 2 == 0:
        num = int((i ** 2) / 2)
        print(num)
    else:
        num = int((i ** 2 - 1) / 2)
        print(num)
