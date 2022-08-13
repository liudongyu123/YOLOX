# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 13、猜年龄.py
@time: 2022/2/3 14:34
"""
for i in range(1, 100):
    for j in range(1, i):
        if i * j == 6 * (i + j) and i - j <= 8:
            print(f"妹妹的年龄为{j}岁")
