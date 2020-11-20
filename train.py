"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    train.py [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--base_model=<base_model>] [--no_cache]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
"""
import os
from docopt import docopt
import sys
# sys.path.append('../donkeycar-linux/')
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import donkeycar as dk

#import parts
# from donkeycar.parts.camera import PiCamera, CSICamera
# from donkeycar.parts.transform import Lambda
from donkeycar.parts.keras import KerasLinear, KerasCategorical
from donkeycar.parts.datastore import TubGroup


def train(cfg, tub_names, new_model_path, base_model_path=None):
    """
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    """
    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    def train_record_transform(record):
        """ convert categorical steering to linear and apply image augmentations """
        record['user/angle'] = dk.util.data.linear_bin(record['user/angle'])
        # TODO add augmentation that doesn't use opencv
        return record

    def val_record_transform(record):
        """ convert categorical steering to linear """
        record['user/angle'] = dk.util.data.linear_bin(record['user/angle'])
        return record

    new_model_path = os.path.expanduser(new_model_path)

    kl = KerasCategorical()
    if base_model_path is not None:
        base_model_path = os.path.expanduser(base_model_path)
        kl.load(base_model_path)

    print('tub_names', tub_names)
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')
    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(
        X_keys,
        y_keys,
        train_record_transform=train_record_transform,
        val_record_transform=val_record_transform,
        batch_size=cfg.BATCH_SIZE,
        train_frac=cfg.TRAIN_TEST_SPLIT)

    total_records = len(tubgroup.df)
    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    kl.train(train_gen,
             val_gen,
             saved_model_path=new_model_path,
             steps=steps_per_epoch,
             train_split=cfg.TRAIN_TEST_SPLIT)


if __name__ == '__main__':
    args = docopt(__doc__)

    cfg = dk.load_config()
    tub = args['--tub']
    new_model_path = args['--model']
    base_model_path = args['--base_model']
    cache = not args['--no_cache']
    train(cfg, tub, new_model_path, base_model_path)
