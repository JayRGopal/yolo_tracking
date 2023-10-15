# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from functools import partial
from pathlib import Path

import csv
import torch
import shutil

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results

# Verify
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(parent_dir)
from utilsVerify import *


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    COLUMN_LABELS = [
        "frame_idx", "timestep", "success", "track_id",
        "nose_x", "nose_y", "nose_conf",
        "left_eye_x", "left_eye_y", "left_eye_conf",
        "right_eye_x", "right_eye_y", "right_eye_conf",
        "left_ear_x", "left_ear_y", "left_ear_conf",
        "right_ear_x", "right_ear_y", "right_ear_conf",
        "left_shoulder_x", "left_shoulder_y", "left_shoulder_conf",
        "right_shoulder_x", "right_shoulder_y", "right_shoulder_conf",
        "left_elbow_x", "left_elbow_y", "left_elbow_conf",
        "right_elbow_x", "right_elbow_y", "right_elbow_conf",
        "left_wrist_x", "left_wrist_y", "left_wrist_conf",
        "right_wrist_x", "right_wrist_y", "right_wrist_conf",
        "left_hip_x", "left_hip_y", "left_hip_conf",
        "right_hip_x", "right_hip_y", "right_hip_conf",
        "left_knee_x", "left_knee_y", "left_knee_conf",
        "right_knee_x", "right_knee_y", "right_knee_conf",
        "left_ankle_x", "left_ankle_y", "left_ankle_conf",
        "right_ankle_x", "right_ankle_y", "right_ankle_conf",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_conf"
    ]

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )


    real_name = os.path.basename(args.source)
    outputs_folder = os.path.join(args.project, real_name)
    if os.path.exists(outputs_folder):
        shutil.rmtree(outputs_folder)
        print(f'Deleted output folder of {real_name} from a previous run!')
    patdata_folder = os.path.join(args.project_patdata, real_name)
    if os.path.exists(patdata_folder):
        shutil.rmtree(patdata_folder)
        print(f'Deleted pose overlay patient data folder of {real_name} from a previous run!')
    os.makedirs(outputs_folder, exist_ok=True)

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project_patdata,
        name=real_name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args

    output_csv_path = os.path.join(outputs_folder, "output.csv")
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the column labels as the header row
        writer.writerow(COLUMN_LABELS)
    

        for frame_idx, r in enumerate(results):

            if args.vid_stride > 1:
                timestep_now = args.vid_stride * (frame_idx + 1 / args.vid_fps)
            else:
                timestep_now = args.vid_stride * (frame_idx / args.vid_fps)
            success = 1

            kp_now = r.keypoints.cpu().numpy().data
            boxes_now = r.boxes.cpu().numpy().data
            num_ppl = kp_now.shape[0]
            for person_id in range(num_ppl):
                # TODO: Add verification here
                orig_img_now = r.orig_img.copy()

                # Save outputs to CSV
                if boxes_now.shape[0] > 0:
                    boxes_person = boxes_now[person_id]
                    track_id_now = int(boxes_person[4])
                    kp_person = kp_now[person_id]
                    tensor_values = [frame_idx, timestep_now, success, track_id_now] + [round(float(value), 3) for value in kp_person.flatten().tolist()]
                    tensor_values = tensor_values + [round(float(value), 3) for value in boxes_person[:4].tolist()]
                    tensor_values = tensor_values + [round(float(boxes_person[5]), 3)]
                    writer.writerow(tensor_values)
                    print(f'Wrote row to {output_csv_path}')
                else:
                    success = 0
                    tensor_values = [frame_idx, timestep_now, success] + [0] * (len(COLUMN_LABELS) - 3)
                    writer.writerow(tensor_values)
                    print(f'Wrote row to {output_csv_path}')
                    break

            if r.boxes.data.shape[1] == 7:

                if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                    p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                    yolo.predictor.mot_txt_path = p
                elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                    p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                    yolo.predictor.mot_txt_path = p

                if args.save_mot:
                    write_mot_results(
                        yolo.predictor.mot_txt_path,
                        r,
                        frame_idx,
                    )

                if args.save_id_crops:
                    for d in r.boxes:
                        print('args.save_id_crops', d.data)
                        save_one_box(
                            d.xyxy,
                            r.orig_img.copy(),
                            file=(
                                yolo.predictor.save_dir / 'crops' /
                                str(int(d.cls.cpu().numpy().item())) /
                                str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                            ),
                            BGR=True
                        )

        if args.save_mot:
            print(f'MOT results saved to {yolo.predictor.mot_txt_path}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--project-patdata', default=ROOT / 'runs' / 'track_PatData',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid-fps', default=30, type=int,
                        help='video frames per second')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
