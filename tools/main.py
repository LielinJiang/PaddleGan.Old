import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import paddle
from ppgan.utils.options import parse_args
from ppgan.utils.config import get_config
from ppgan.engine.trainer import Trainer


def main(args, cfg):
    paddle.enable_imperative()

    trainer = Trainer(cfg)

    # continue train or evaluate, checkpoint need contain epoch and optimizer info
    if args.resume:
        trainer.resume(args.resume)
    # evaluate or finute, only load generator weights
    elif args.load:
        trainer.load(args.load)

    if args.evaluate_only:
        trainer.test()
        return

    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.config_file)

    if args.evaluate_only:
        cfg.isTrain = False

    print(cfg, cfg.model.name, cfg.isTrain)
    main(args, cfg)
    