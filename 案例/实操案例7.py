# # -*- coding: utf-8 -*-
#
# """
# @author: l'd'y
# @file: 实操案例7.py
# @time: 2022/1/28 13:10
# """
# c = ["白羊座", "金牛座"]
# n = ["积极乐观", "性格内向"]
# # d = zip(c, n)   # 元组
# d = dict(zip(c, n))
# # for i in d:
# #     print(i)
# print(d)
# key = input("请输入您要查询的星座：")
# flag = True
# for i in d:
#     if key == i:
#         print(key, "的特点是", d.get(key))
#         flag = True
#         break
#     else:
#         flag = False
#         # print("输入有误")
# if flag == False:
#     print("输入有误")
c = ["白羊座", "金牛座"]
n = ["积极乐观", "性格内向"]
# d = zip(c, n)   # 元组
d = dict(zip(c, n))
print(d)
print(d.keys())
print(d.values())
print(d.items())
