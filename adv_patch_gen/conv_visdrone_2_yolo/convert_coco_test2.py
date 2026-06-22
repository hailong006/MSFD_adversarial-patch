import json
import os
import xml.etree.ElementTree as ET

# 这个文件可以把input_folder下的文件夹的xml文件全部转换成coco类型，并且输出到output_folder文件夹下
# 如果input_folder下的xml有多个类别，都可以挑出来

# 修改输入和输出文件夹的路径
input_folder = "E:\\0.4Program-Code\\3.YOLOv5-uav-adversarial\\yolov5_adversarial-master\\yolov5_adversarial-master\\adv_patch_gen\\conv_visdrone_2_yolo\\xml"
output_folder = "E:\\0.4Program-Code\\3.YOLOv5-uav-adversarial\\yolov5_adversarial-master\\yolov5_adversarial-master\\adv_patch_gen\\conv_visdrone_2_yolo\\coco"
# 列出输入文件夹中的所有文件
files = os.listdir(input_folder)

# 遍历所有文件并检查扩展名以仅处理 XML 文件
for file in files:
    if file.endswith(".xml"):
        # 读取原始标注 XML 文件
        with open(os.path.join(input_folder, file), "r") as f:
            xml_data = f.read()

        # 解析 XML 数据
        root = ET.fromstring(xml_data)

        # 获取图像路径
#        path = root.find("path").text

        # 获取图像尺寸
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        if width == 0 or height == 0:
            print(f"Invalid width or height in file: {file}")
            continue

        # 获取目标位置
        objects = root.findall("object")
        shapes = []

        for obj in objects:
            xmin = int(obj.find("bndbox/xmin").text)
            #xmin = int(float(obj.find("bndbox/xmin").text))
            ymin = int(obj.find("bndbox/ymin").text)
            #ymin = int(float(obj.find("bndbox/ymin").text))
            xmax = int(obj.find("bndbox/xmax").text)
            #xmax = int(float(obj.find("bndbox/xmax").text))
            ymax = int(obj.find("bndbox/ymax").text)
            #ymax = int(float(obj.find("bndbox/ymax").text))

            # 转换目标位置坐标为 COCO 格式
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            bbox_width = (xmax - xmin)
            bbox_height = (ymax - ymin)

            # 读取目标标签
            label = obj.find("name").text

            # 生成 COCO 格式的 JSON
            shape = {
                "label": label,
                "points": [
                    [x_center - bbox_width / 2, y_center - bbox_height / 2],
                    [x_center + bbox_width / 2, y_center + bbox_height / 2],
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)

        # 生成最终的 COCO 格式的 JSON 数据
        coco_data = {
            "version": "5.1.1",
            "flags": {},
            "imagePath": None,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "shapes": shapes
        }

        # 创建唯一的输出文件名
        output_file = os.path.splitext(file)[0] + ".json"

        # 将数据保存为 JSON 文件
        with open(os.path.join(output_folder, output_file), "w") as f:
            json.dump(coco_data, f, indent=4)

