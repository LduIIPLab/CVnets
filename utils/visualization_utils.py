from torch import Tensor
import cv2
import numpy as np
import copy
from typing import Optional, List, Union, Tuple
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

from utils.color_map import Colormap
from utils import logger

# -------- cam -------------

from torchvision import models
import os
import glob
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise

from options.opts import get_training_arguments
from cvnets import get_model

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

FONT_SIZE = cv2.FONT_HERSHEY_PLAIN
LABEL_COLOR = [255, 255, 255]
TEXT_THICKNESS = 1
RECT_BORDER_THICKNESS = 2


def plot_list(x_list: list,
              y_list: list,
              label_list: list,
              x_label: str,
              y_label: str,
              x_limit: Optional[tuple] = (0, 100),
              y_limit: Optional[tuple] = (0, 100),
              color_list: Optional[tuple] = ('g', 'c', 'b', 'r'),
              marker_list: Optional[tuple] = ('o', '+', 'x', 'd'),
              line_style_list: Optional[tuple] = ('dotted', 'dashed', 'dashdot', 'solid'),
              legend_loc: Optional[str] = "lower right",
              legend_font_size: Optional[Union[str, int]] = "x-large",
              axes_label_font_size: Optional[Union[str, int]] = "x-large",
              *args, **kwargs):

    for idx in range(len(x_list)):
        plt.plot(x_list[idx], y_list[idx],
                 color=color_list[idx],
                 marker=marker_list[idx],
                 linestyle=line_style_list[idx],
                 label=label_list[idx])

    plt.xlim(x_limit)
    plt.ylim(y_limit)

    plt.legend(loc=legend_loc, fontsize=legend_font_size)
    plt.grid()
    plt.xlabel(xlabel=x_label, fontsize=axes_label_font_size)
    plt.ylabel(ylabel=y_label, fontsize=axes_label_font_size)

    plt.show()


# -------- cam -------------


def plot_cam(
        target_layer_list: Optional[list] = None,
        size: Optional[Tuple] = None,
        image_path: Optional[str] = r'.\test_images\*.jpg',
        cam_path: Optional[str] = r".\cam_results",
):
    for i in glob.glob(image_path):
        print(i)

    print(cam_path)
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    args = lambda: None
    args.use_cuda = False
    args.method = "gradcam"
    args.eigen_smooth = True
    args.aug_smooth = True

    opts = get_training_arguments()
    model = get_model(opts)
    target_layers = [model.layer_5[-1]] if target_layer_list is None else target_layer_list
    targets = None

    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:
        for file_name in glob.glob(image_path):
            rgb_img = cv2.imread(file_name, 1)[:, :, ::-1]
            if size is not None:
                rgb_img = cv2.resize(rgb_img, size)
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
            gb = gb_model(input_tensor, target_category=None)

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)

            file_path = cam_path + "\\" + os.path.split(file_name)[-1]
            cv2.imwrite(file_path, cam_image)


def concat_images(
        image_path: Optional[str] = r'.\cam_results\*.jpg',
        size: Optional[Tuple] = None,
        num_column: Optional[int] = 10,
        padding: Optional[Union[int, tuple]] = None
):
    if padding is not None:
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple) and len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        elif isinstance(padding, tuple) and len(padding) != 4:
            raise ValueError("padding must be an integer or a tuple with length 2 or 4.")

    cnt = -1
    col_images = []
    for file_name in glob.glob(image_path):
        cnt += 1
        if cnt % num_column == 0:
            row_imgs = []
        img = cv2.imread(file_name)
        if size is not None:
            img = cv2.resize(img, size)
        if padding:
            # BORDER_CONSTANT BORDER_REFLECT BORDER_DEFAULT BORDER_REPLICATE BORDER_WRAP
            img = cv2.copyMakeBorder(img, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT,
                                     value=(255, 255, 255))
        row_imgs.append(img)
        if cnt % num_column == num_column - 1:
            col_images.append(cv2.hconcat(row_imgs))

    img_rel = cv2.vconcat(col_images)
    rel_path = os.path.split(image_path)[0] + "\\combine_cam\\" + "combine_cam.jpg"
    cv2.imwrite(rel_path, img_rel)
