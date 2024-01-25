from utils.visualization_utils import plot_cam, concat_images
import sys


def cam_dataset_model(sel_dataset, sel_model, n_classes, size=None, image_path: str = None):
    sys.argv.append('--common.config-file')
    sys.argv.append('config/classification/food_image/ehfr_net_food101.yaml'.format(sel_dataset))

    sys.argv.append('--model.classification.pretrained')
    sys.argv.append(r'.\cam_relative_file\{}\{}\checkpoint_ema_best.pt'.format(sel_dataset, sel_model))

    sys.argv.append('--common.override-kwargs')
    # sys.argv.append('model.classification.msv2.width_multiplier=0.5')
    sys.argv.append('model.classification.n_classes={}'.format(n_classes))

    plot_cam(image_path=r'.\cam_relative_file\{}\origin\*.jpg'.format(sel_dataset),
             cam_path=r'.\cam_relative_file\{}\{}\cam_results'.format(sel_dataset, sel_model), size=size)


def concatenate_images(
        sel_dataset,
        image_path=r'D:\cam_images\{}\total\*.jpg',
        size=None,
        padding=None,
        num_column=4,
):
    concat_images(image_path=image_path.format(sel_dataset), num_column=num_column, size=size, padding=padding)


# generative heat map
cam_dataset_model(sel_dataset="food101", sel_model="msfvit", n_classes="101", size=(256, 256))

# concatenate images
# concatenate_images(sel_dataset="food172", image_path=r'.\cam_relative_file\{}\total\*.jpg', size=None,
#                    padding=10, num_column=2)
