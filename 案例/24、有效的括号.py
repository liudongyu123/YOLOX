# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 24、有效的括号.py
@time: 2022/2/15 20:13
"""


# 方法一：字符串替换法

# def valid_str(string):
#     if len(string) % 2 == 1:
#         return False
#     while "()" in string or "[]" in string or "{}" in string:
#         string = string.replace("()", "")
#         string = string.replace("[]", "")
#         string = string.replace("{}", "")
#         return string == ""
#
#
# print(valid_str("()[]{[())]}"))
# 方法二：栈
def valid_str(string):
    if len(string) % 2 == 1:
        return False
    stack = []
    char_dict = {
        ")": "(",
        "}": "{",
        "]": "["
    }
    for char in string:
        if char in char_dict:
            if not stack or char_dict[char] != stack.pop():
                return False
        else:
            stack.append(char)
    return not stack


print(valid_str("()"))
print(valid_str("()[]{[())]}"))