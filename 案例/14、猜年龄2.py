# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 14、猜年龄2.py
@time: 2022/2/3 14:45
"""
for i in range(10, 40):

    q1 = (str(i ** 3))
    q2 = (str(i ** 4))
    if len(q1) == 4 and len(q2) == 6:
        # print(i)
        if len(set(q1 + q2)) == 10:
            print(i)
