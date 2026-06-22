import os
import json

##这个文件是把转换后的json文件的img路径替换为自己想要的路径

# 输入、输出文件夹路径和待添加的路径
input_folder = "E:\\0.4Program-Code\\3.YOLOv5-uav-adversarial\\yolov5_adversarial-master\\yolov5_adversarial-master\\adv_patch_gen\\conv_visdrone_2_yolo\\coco"
output_folder = "D:\\Open-mmLab\\yolov5-master\\classify\\ann_savepath"
prefix = "D:\\Open-mmLab\\yolov5-master\\classify\\img_savepath\\"
# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的json文件
for filename in os.listdir(input_folder):
    # 判断是否为json文件
    if filename.endswith(".json"):
        # 读取json文件内容
        input_path = os.path.join(input_folder, filename)
        with open(input_path, "r") as f:
            data = json.load(f)
        # 修改imagePath字段值
        image_path = prefix + filename[:-5] + ".jpg"  # 构造新的imagePath路径
        data["imagePath"] = image_path
        # 保存修改后的内容到输出文件夹
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

