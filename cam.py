from utils.visualization_utils import plot_cam, concat_images
import sys


def cam_dataset_model(sel_dataset, sel_model, n_classes, size=None):
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
        image_path=r'.\cam_relative_file\{}\total\*.jpg',
        size=None,
        padding=None,
        num_column=4,
):
    concat_images(image_path=image_path.format(sel_dataset), num_column=num_column, size=size, padding=padding)


'''
    Before running the code, it is necessary to follow the CAM_ Relative_ Create a folder in the form of a file and 
    place it in the trained model file.
'''

# generative heat map
# sel_dataset: The name of the folder where the dataset is stored.
# sel_model: Model name.
# n_classes: Number of categories.
# size: Enter image size.
cam_dataset_model(sel_dataset="food101", sel_model="msfvit", n_classes="101", size=(256, 256))

# concatenate images
# sel_dataset: The name of the folder where the dataset is stored.
# image_path: The position of the images that need to be spliced.
# concatenate_images(sel_dataset="food172", image_path=r'.\cam_relative_file\{}\total\*.jpg', size=None,
#                    padding=10, num_column=2)
