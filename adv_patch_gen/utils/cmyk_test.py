# import os
# def rgb_to_cmyk(r, g, b):
#     # 将RGB值转换为0-1范围内的百分比
#     r = r / 255.0
#     g = g / 255.0
#     b = b / 255.0

#     # 计算CMY值
#     c = 1 - r
#     m = 1 - g
#     y = 1 - b

#     # 计算K值（黑色墨水量）
#     k = min(c, m, y)

#     # 如果K接近于1，说明颜色接近黑色，直接返回CMYK值
#     if k == 1:
#         return (0, 0, 0, 100)
#     else:
#         # 根据K值调整CMY值
#         c = (c - k) / (1 - k)
#         m = (m - k) / (1 - k)
#         y = (y - k) / (1 - k)

#         # 将CMY值和K值转换为0-100范围内的百分比
#         c = round(c * 100)
#         m = round(m * 100)
#         y = round(y * 100)
#         k = round(k * 100)

#         return (c, m, y, k)

# def cmyk_to_rgb(c, m, y, k):
#     # 将CMYK值转换为0-1范围内的百分比
#     c = c / 100.0
#     m = m / 100.0
#     y = y / 100.0
#     k = k / 100.0

#     # 计算RGB值
#     r = 255 * (1 - c) * (1 - k)
#     g = 255 * (1 - m) * (1 - k)
#     b = 255 * (1 - y) * (1 - k)

#     # 确保RGB值在0-255的范围内
#     r = min(max(0, r), 255)
#     g = min(max(0, g), 255)
#     b = min(max(0, b), 255)

#     return (int(r), int(g), int(b))

# # 示例：将RGB颜色(128, 64, 192)转换为CMYK颜色
# rgb_color = (128, 64, 192)
# cmyk_color = rgb_to_cmyk(*rgb_color)
# print(f"CMYK颜色: {cmyk_color}")

# # 将CMYK颜色转换回RGB颜色
# rgb_color_again = cmyk_to_rgb(*cmyk_color)
# print(f"转换回的RGB颜色: {rgb_color_again}")
'''
RGB_2>LAB
import numpy as np
from skimage import color

def rgb_to_lab(r, g, b):
    # 将 RGB 值规范化到 0-1 的范围
    rgb = np.array([r, g, b]) / 255.0
    
    # 将 RGB 转换为 Lab
    lab = color.rgb2lab(rgb.reshape(1, 1, 3))
    
    # 提取 Lab 值
    L, a, b = lab[0][0]
    
    return L, a, b

# 示例
r, g, b = 255, 0, 0  # 红色
L, a, b = rgb_to_lab(r, g, b)
print(f'Lab 值: L = {L:.2f}, a = {a:.2f}, b = {b:.2f}')
'''
import numpy as np

def rgb_to_cmyk(r, g, b):
    """将RGB值转换为CMYK值"""
    # 将RGB值归一化到[0, 1]
    r, g, b = [x / 255.0 for x in (r, g, b)]
    
    # 计算K值
    k = 1 - max(r, g, b)
    if k < 1:
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)
    else:
        c = m = y = 0

    return (c, m, y, k)

def rgb_to_lab(r, g, b):
    """将RGB值转换为LAB值"""
    # 将RGB值归一化到[0, 1]
    r, g, b = [x / 255.0 for x in (r, g, b)]
    
    # 处理sRGB色彩空间
    r = r ** (1/2.2)
    g = g ** (1/2.2)
    b = b ** (1/2.2)

    # 转换为XYZ
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # 转换为LAB
    xyz_ref = (95.047, 100.000, 108.883)  # D65白点
    x /= xyz_ref[0]
    y /= xyz_ref[1]
    z /= xyz_ref[2]

    # 照明补偿
    for i in (x, y, z):
        if i > 0.008856:
            i = i ** (1/3)
        else:
            i = (i * 7.787) + (16 / 116)

    L = max(0, min(100, (116 * y) - 16))
    a = max(-128, min(127, (x - y) * 500))
    b = max(-128, min(127, (y - z) * 200))

    return (L, a, b)

def lab_distance(lab1, lab2):
    """计算两个LAB值之间的欧氏距离"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))

# 示例输入
r = 255
g = 0
b = 0

# 第一步：RGB转CMYK
cmyk = rgb_to_cmyk(r, g, b)
print("CMYK:", cmyk)

# 第二步：RGB转LAB
lab_from_rgb = rgb_to_lab(r, g, b)
print("LAB from RGB:", lab_from_rgb)

# 第三步：CMYK转RGB，然后转换为LAB
# 将CMYK转换回RGB（为便于计算，假设K为0）
c, m, y, k = cmyk
r = 255 * (1 - c) * (1 - k)
g = 255 * (1 - m) * (1 - k)
b = 255 * (1 - y) * (1 - k)
lab_from_cmyk = rgb_to_lab(r, g, b)
print("LAB from CMYK:", lab_from_cmyk)

# 第四步：计算LAB值之间的欧氏距离
distance = lab_distance(lab_from_rgb, lab_from_cmyk)
print("Euclidean distance between LAB values:", distance)
