import sys
from typing import Optional


def ehfr_net_config_forward(
        dataset_name: str = 'food101',
        width_multiplier: float = 0.5,
        is_eval: Optional[bool] = False,
        is_test: Optional[bool] = False,
        task_name: str = None,
):

    num_classes = int(dataset_name[-3:])

    sys.argv.append('--common.config-file')
    sys.argv.append('config/classification/food_image/ehfr_net_{}.yaml'.format(dataset_name))

    sys.argv.append('--common.results-loc')
    sys.argv.append('{}_results'.format(dataset_name))

    if task_name:
      sys.argv.append('--common.run-label')
      sys.argv.append('ehfr_net_{}_width_multiplier_{}'.format(task_name, width_multiplier))
    else:
      sys.argv.append('--common.run-label')
      sys.argv.append('ehfr_net_width_multiplier_{}'.format(width_multiplier))

    sys.argv.append('--common.override-kwargs')
    sys.argv.append('model.classification.ehfr_net.width_multiplier={}'.format(width_multiplier))

    if is_eval:
        sys.argv.append('--model.classification.pretrained')
        sys.argv.append('/{}_results/ehfr_net_width_multiplier_{}/checkpoint_ema_best.pt'.format(
            dataset_name, width_multiplier))

    if is_test:
        sys.argv.append('model.classification.n_classes={}'.format(num_classes))


