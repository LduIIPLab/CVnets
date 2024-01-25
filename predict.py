import glob

import torch
import sys
import os
import json
import argparse
from PIL import Image

from cvnets import get_model
from options.opts import get_training_arguments
from torchvision import transforms as T


def create_image_classes_dict(data_path):
    assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

    image_class = [cla for cla in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cla))]

    image_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(image_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('./image_classes_dict/class_indices.json', 'w') as json_file:
        json_file.write(json_str)


def set_model_argument():
    sys.argv.append('--common.config-file')
    sys.argv.append('config/classification/food_image/ehfr_net_food101.yaml')

    sys.argv.append('--model.classification.n-classes')
    sys.argv.append('101')


def set_args(image_path: str = None):
    # set the device
    sys.argv.append('--use-cuda')

    # set the path that is used to analysis
    if image_path:
        sys.argv.append('--image-path')
        sys.argv.append(image_path)
    else:
        sys.argv.append('--image-path')
        sys.argv.append(r'.\cam_relative_file\food101\origin\*.jpg')

    # set the weights path
    sys.argv.append('--weights_path')
    sys.argv.append(r'.\cam_relative_file\food101\ehfr_net\checkpoint_ema_best.pt')


def get_args_other():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')

    parser.add_argument('--weights_path', type=str, default=None, help='Input weights path')
    parser.add_argument('--common.config-file', type=str, default=None, help='Test')
    parser.add_argument('--model.classification.n-classes', type=int, default=None, help='the number of classification')

    args = parser.parse_args()

    return args


def get_image_name(path_org):
    name = os.path.basename(path_org)
    return name


def predict(image_path, model, data_transform, device):
    img = Image.open(image_path)

    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    json_path = './image_classes_dict/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)

    max_prob = 0
    class_name = 0
    for i in range(len(predict)):
        if i == 0:
            class_name = class_indict[str(i)]
            max_prob = predict[i].numpy()
        elif predict[i].numpy() > max_prob:
            class_name = class_indict[str(i)]
            max_prob = predict[i].numpy()

    image_name = get_image_name(image_path)
    print("image: {} The most likely species: {:10}   it's prob: {:.3}".format(image_name, class_name, max_prob))


def predict_run(model):
    set_args()
    opts = get_args_other()

    img_size = 256
    data_transform = T.Compose(
        [T.Resize(size=288, interpolation=Image.BICUBIC),
         T.CenterCrop(img_size),
         T.ToTensor()])

    if opts.use_cuda and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = model.to(device)

    model.load_state_dict(torch.load(opts.weights_path, map_location=device))
    for image_name in glob.glob(opts.image_path):
        predict(image_path=image_name, model=model, data_transform=data_transform, device=device)


def setup_model():
    set_model_argument()
    opts = get_training_arguments()

    # set-up the model
    model = get_model(opts)

    set_args()
    opts = get_args_other()

    if opts.use_cuda and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = model.to(device)

    model.load_state_dict(torch.load(opts.weights_path, map_location=device))

    return model, device


def main():
    # This needs to be replaced with the path where the data set is located
    data_path = r'/ai35/Food/Food-101'
    classes_json_path = './image_classes_dict/class_indices.json'
    if os.path.exists(classes_json_path):
        pass
    else:
        create_image_classes_dict(data_path)
    # set_model_argument()
    set_model_argument()
    opts = get_training_arguments()

    # set-up the model
    model = get_model(opts)
    # print(model)
    predict_run(model=model)


if __name__ == '__main__':
    main()
