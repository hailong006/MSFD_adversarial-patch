# # 生成nosiy图像
# from PIL import Image
# import numpy as np
# import random
# import matplotlib.pyplot as plt

# def add_noise(image_path, noise_level=25):
#     # 打开图片
#     img = Image.open(image_path)
#     # 转换为数组
#     img_array = np.array(img)
    
#     # 生成噪声
#     noise = np.random.randint(-noise_level, noise_level + 1, img_array.shape, dtype=np.int16)
    
#     # 添加噪声到图片
#     noisy_img_array = img_array + noise
    
#     # 确保值在0到255之间
#     noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    
#     # 转换回图片
#     noisy_img = Image.fromarray(noisy_img_array)
    
#     return noisy_img

# # 使用示例
# input_image_path = '2.jpg'  # 替换为您的输入图片路径
# noisy_image = add_noise(input_image_path, noise_level=50)

# # 显示原图和加噪后的图像
# original_image = Image.open(input_image_path)

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.title("原始图像")
# plt.imshow(original_image)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("添加噪声后的图像")
# plt.imshow(noisy_image)
# plt.axis('off')

# plt.show()


#中值滤波
from PIL import Image
import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

def median_filter_image(image_path, size=3):
    # 打开图片
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # 对每个颜色通道分别进行中值滤波
    filtered_img_array = np.zeros_like(img_array)
    for channel in range(img_array.shape[2]):
        filtered_img_array[:, :, channel] = median_filter(img_array[:, :, channel], size=size)
    
    # 转换回图片
    filtered_img = Image.fromarray(filtered_img_array)
    
    return filtered_img, img

# 使用示例
input_image_path = 'noise_patch.png'  # 替换为您的输入图片路径
filtered_image, original_image = median_filter_image(input_image_path, size=4)
#filtered_image, original_image = median_filter_image(filtered_image, size=3)
# 显示原图和滤波后的图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("原始图像")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("中值滤波后的图像")
plt.imshow(filtered_image)
plt.axis('off')

plt.show()