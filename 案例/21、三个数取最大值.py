# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 21、三个数取最大值.py
@time: 2022/2/15 19:38
"""
a, b, c = 10, 60, 18
if a > b:
    max_num = a
else:
    max_num = b
if max_num < c:
    max_num = c
print("最大值是：", max_num)
