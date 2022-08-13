# -*- coding: utf-8 -*-

"""
@author: l'd'y
@file: 11、剪刀石头布.py
@time: 2022/2/3 13:41
"""
import random

print("1代表剪刀、2代表石头、3代表布")
score = 100
game_info = {1: "剪刀", 2: "石头", 3: "布"}
while True:
    robots_choice = random.randint(1, 3)
    user_choice = input("请出拳：")
    if user_choice not in "123":
        print("输入错误，请重新出拳：")
        continue
    user_choice = int(user_choice)
    print(f"电脑出{game_info[robots_choice]}")
    print(f"你出{game_info[user_choice]}")
    if user_choice == 1 and robots_choice == 3 \
            or user_choice == 2 and robots_choice == 1 \
            or user_choice == 3 and robots_choice == 2:
        print("本局游戏你赢了")
        score += 10
        print(f"当前分数：{score}")
    elif user_choice == robots_choice:
        print("本轮游戏平局")
        print(f"当前分数：{score}")
    else:
        print("本局游戏你输了")
        score -= 10
        print(f"当前分数：{score}")
    if score >= 200:
        print("游戏结束、你赢了")
    if score <= 0:
        print("游戏结束、你输了")
