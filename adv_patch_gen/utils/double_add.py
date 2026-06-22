import cv2
import numpy as np

# 1. 读取原补丁
original_patch = cv2.imread('original_patch.png')

# 2. 获取原补丁的尺寸
height, width = original_patch.shape[:2]

# 3. 生成新的补丁 (尺寸是原补丁的一半)
new_height, new_width = height // 2, width // 2
new_patch = np.zeros((new_height, new_width, 3), dtype=np.uint8)

# （这里你可以填充 new_patch，可以是随机颜色，或其他逻辑）

# 4. 计算新补丁的放置位置
y_offset = (height - new_height) // 2
x_offset = (width - new_width) // 2

# 5. 叠加新补丁到原补丁的中心
# 确保叠加的区域在原补丁的范围内
original_patch[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = new_patch

# 6. 保存或显示结果
cv2.imwrite('combined_patch.png', original_patch)
cv2.imshow('Combined Patch', original_patch)
cv2.waitKey(0)
cv2.destroyAllWindows()
