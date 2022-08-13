# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 12、快乐数.py
@time: 2022/2/3 14:02
"""


def sum_square(n):
    sum_num = 0
    for i in str(n):
        sum_num += int(i) ** 2
    return sum_num


while True:
    list1 = []
    # n = int(input("请输入一个数:"))
    for n in range(18, 20):
        while sum_square(n) not in list1:
            list1.append(n)
            j = n
            print(j)
            n = sum_square(n)

    # if n == 1:
    #     print(f"{j}是一个快乐数")
    #     continue

        # else:
        #     print("不是一个快乐数")
