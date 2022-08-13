# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: python书.py
@time: 2022/3/4 10:50
"""
# 用字典判断字符串出现的次数
# a = "life is short i study python"
# a = a.replace(" ", "")
# dict1 = dict()
# for i in a:
#     if i in dict1:
#         dict1[i] += 1
#     else:
#         dict1[i] = 1
# for k in dict1:
#     print(k, dict1[k])
# 用户登陆系统
users = {"张三": "123456", "李四": "234567", "王五": "345678"}
count = 3
while True:
    print("*" * 40)
    print("欢迎登录系统！")
    name = input("请输入要登陆的用户名：")
    if name in users:
        while count >= 1:
            password = input("请输入密码：")
            if users[name] == password:
                print("登录成功！")
                break
            else:
                count -= 1
                print(f"密码输入错误，您还有{count}次机会")

        else:
            print("您的机会已用完！")
        break
    else:
        flag = input("用户名不存在！\n 是否创建用户[Y/N]:")
        if flag == "y" or flag == "Y":
            while True:
                name = input("请输入创建的用户名：")
                if name in users:
                    print("输入的用户名已经存在，请重新输入：")
                    break
                password1 = input("请设置密码：")
                password2 = input("请确认密码：")
                if password2 == password1:
                    users[name] = password1
                    print("用户创建成功！")
                    break
                else:
                    print("两次密码输入不一致，请重新输入：")
                    break
        else:
            print("您输入错误，不创建新用户")
            continue

    # else:
    #     print("欢迎再次使用此系统！再见！")
    #     break
