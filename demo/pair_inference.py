from pathlib import Path
import yaml
from detectron2.data.datasets import register_coco_instances

# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import sys

from vlpart.evaluation.paco_evaluation import PACOEvaluator

sys.path.append(".")
from vlpart.config import add_vlpart_config

from predictor import PairVisualizationDemo

# constants
WINDOW_NAME = "image demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_vlpart_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )

    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 530
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument(
    #     "--webcam", action="store_true", help="Take inputs from webcam."
    # )
    # parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="pascal_part",
        choices=[
            "pascal_part",
            "partimagenet",
            "paco",
            "voc",
            "coco",
            "lvis",
            "pascal_part_voc",
            "lvis_paco",
            "custom",
        ],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    with open(Path(__file__).parent / "pair_cfg.yaml") as f:
        paths = yaml.safe_load(f)

    json_annotation_val = paths["json_annotation_val_path"]
    imgs_dir = paths["imgs_dir"]
    register_coco_instances("paco_pair_val", {}, json_annotation_val, imgs_dir)

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    vocab_path = paths["vocab_path"]
    with open(vocab_path) as f:
        vocab = json.load(f)

    args.custom_vocabulary = ",".join(vocab)

    cfg = setup_cfg(args)
    demo = PairVisualizationDemo(cfg, args)

    # evaluator = COCOEvaluator("paco_pair_val", output_dir="./output")
    evaluator = PACOEvaluator("paco_pair_val", cfg, True, "./output", False, False)
    val_loader = build_detection_test_loader(cfg, "paco_pair_val")
    print(inference_on_dataset(demo.predictor.model, val_loader, evaluator))
