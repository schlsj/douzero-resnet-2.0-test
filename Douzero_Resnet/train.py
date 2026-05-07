import os

from douzero.dmc import parser
from douzero.dmc.dmc import train
from douzero.dmc.device_utils import resolve_training_device

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.training_device = resolve_training_device(flags.training_device)
    if flags.training_device != 'tpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
