import os
import json
import random
import shutil

# json转成yolo格式，划分的时候把json和img全部放入input_dir里面
# 先用xml转json,再用json转为yolo格式
# 定义你的类别列表
classes = [
    'ignored regions',
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning-tricycle',
    'bus',
    'motor',
    'others'
]  # 更新为你的类别


def convert_annotation(input_dir, output_dir, json_file):
    with open(os.path.join(input_dir, json_file), 'r') as in_file:
        data = json.load(in_file)

    out_file = open(os.path.join(output_dir, json_file.replace('.json', '.txt')), 'w')

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    for obj in data['shapes']:
        cls = obj['label']
        if cls not in classes:
            continue

        cls_id = classes.index(cls)
        points = obj['points']
        x_center = (points[0][0] + points[1][0]) / 2.0 / img_width
        y_center = (points[0][1] + points[1][1]) / 2.0 / img_height
        width = (points[1][0] - points[0][0]) / img_width
        height = (points[1][1] - points[0][1]) / img_height

        out_file.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


def convert_all(input_dir, output_dir):
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    random.shuffle(json_files)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_output_dir = os.path.join(output_dir, 'train')
    val_output_dir = os.path.join(output_dir, 'val')
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    val_images_dir = os.path.join(output_dir, 'images', 'val')

    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)

    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    if not os.path.exists(train_images_dir):
        os.makedirs(train_images_dir)

    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir)

    train_split = int(len(json_files) * 0.9)
    train_json_files = json_files[:train_split]
    val_json_files = json_files[train_split:]

    for json_file in train_json_files:
        shutil.copy(os.path.join(input_dir, json_file), train_output_dir)
        convert_annotation(input_dir, train_output_dir, json_file)
        if os.path.exists(os.path.join(input_dir, json_file.replace('.json', '.jpg'))):
            shutil.copy(os.path.join(input_dir, json_file.replace('.json', '.jpg')), train_images_dir)
        elif os.path.exists(os.path.join(input_dir, json_file.replace('.json', '.png'))):
            shutil.copy(os.path.join(input_dir, json_file.replace('.json', '.png')), train_images_dir)

    for json_file in val_json_files:
        shutil.copy(os.path.join(input_dir, json_file), val_output_dir)
        convert_annotation(input_dir, val_output_dir, json_file)
        if os.path.exists(os.path.join(input_dir, json_file.replace('.json', '.jpg'))):
            shutil.copy(os.path.join(input_dir, json_file.replace('.json', '.jpg')), val_images_dir)
        elif os.path.exists(os.path.join(input_dir, json_file.replace('.json', '.png'))):
            shutil.copy(os.path.join(input_dir, json_file.replace('.json', '.png')), val_images_dir)

    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for cls in classes:
            f.write(cls + '\n')


if __name__ == "__main__":
    input_dir = 'E:\\0.4Program-Code\\3.YOLOv5-uav-adversarial\\yolov5_adversarial-master\\yolov5_adversarial-master\\adv_patch_gen\\conv_visdrone_2_yolo\\coco'
    output_dir = 'E:\\0.4Program-Code\\3.YOLOv5-uav-adversarial\\yolov5_adversarial-master\\yolov5_adversarial-master\\adv_patch_gen\\conv_visdrone_2_yolo\\yolo_format'
    convert_all(input_dir, output_dir)

