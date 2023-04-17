import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from collections import defaultdict


import torch
from models.experimental import attempt_load
from utils.torch_utils import TracedModel, select_device

# device = select_device('cpu')  # or 'cpu'
# COMMON_DETECT_MODEL = attempt_load("/Users/lixiaolong/manyProjects/githubProjects/yolov7-on-nvidia-orin/yolov7.pt", map_location=device)  # load FP32 model

# 以cpu的方式运行yolo
# device = select_device('cpu')  # or 'cpu'
# COMMON_DETECT_MODEL = attempt_load("/app/yolov7.pt", map_location=device)  # load FP32 model

# 以gpu的方式运行yolo
# GPU_DEVICE = select_device('cpu')
# COMMON_DETECT_MODEL = attempt_load("/root/yolo/yolonv7-on-nvidia-orin/yolov7.pt", map_location=GPU_DEVICE)  # load FP32 model

GPU_DEVICE = select_device('0')
COMMON_DETECT_MODEL = attempt_load("/app/yolov7.pt", map_location=GPU_DEVICE)  # load FP32 model


def common_detect(
        weights: str = 'yolov7.pt',
        source: str = 'inference/images',
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = '',
        view_img: bool = False,
        save_txt: bool = False,
        classes: int = None,
        agnostic_nms: bool = False,
        augment: bool = False,
        project: str = 'runs/detect',
        name: str = 'exp',
        exist_ok: bool = False,
        no_trace: bool = True,
):
    global GPU_DEVICE
    global COMMON_DETECT_MODEL
    source, weights, view_img, save_txt, imgsz, trace = source, weights, view_img, save_txt, img_size, not no_trace
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()

    # device = select_device(device)
    half = GPU_DEVICE.type != 'cpu'  # half precision only supported on CUDA

    # Load model: 启动时加载
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(COMMON_DETECT_MODEL.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        COMMON_DETECT_MODEL = TracedModel(COMMON_DETECT_MODEL, GPU_DEVICE, img_size)

    if half:
        COMMON_DETECT_MODEL.half()  # to FP16

    # Set Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = COMMON_DETECT_MODEL.module.names if hasattr(COMMON_DETECT_MODEL, 'module') else COMMON_DETECT_MODEL.names

    # Run inference
    if GPU_DEVICE.type != 'cpu':
        COMMON_DETECT_MODEL(torch.zeros(1, 3, imgsz, imgsz).to(GPU_DEVICE).type_as(next(COMMON_DETECT_MODEL.parameters())))  # run once

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(GPU_DEVICE)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if GPU_DEVICE.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                COMMON_DETECT_MODEL(img, augment=augment)[0]

        # Inference
        t1_a = time.time()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = COMMON_DETECT_MODEL(img, augment=augment)[0]
        t1_b = time.time()
        print(f"cost 1: {t1_b - t1_a} s")

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t1_c = time.time()
        print(f"cost 2: {t1_c - t1_b} s")

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=GPU_DEVICE)['model']).to(GPU_DEVICE).eval()

            # Apply Classifier
            pred = apply_classifier(pred, modelc, img, im0s)

        names_obj = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

        # Process detections
        result_dic = defaultdict(lambda: {"items": [], "nums": 0})  # 最终结果

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names_obj[c]
                    box_data = {'confidence': float(conf), "x": float(xyxy[0]), "y": float(xyxy[1]),
                                "width": float(xyxy[2] - xyxy[0]), "height": float(xyxy[3] - xyxy[1])}

                    result_dic[label]['items'].append(box_data)
                    result_dic[label]['nums'] += 1

    print(f'Done. ({time.time() - t0:.3f}s)')
    return result_dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

if __name__ == '__main__':
    common_detect(source="/app/inference/images", no_trace=True)
