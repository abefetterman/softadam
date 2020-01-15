import math
import types

import torch
import torch.optim
from fairseq.optim import FairseqOptimizer, register_optimizer
from .softadam import SoftAdam

@register_optimizer('softadam')
class FairseqRAdam(FairseqOptimizer):

    def __init__(self, args, params):
        super().__init__(args)

        self._optimizer = SoftAdam(params, **self.optimizer_config)
        self._optimizer.name = args.tb_tag + '_softadam'

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--tb-tag', default="", type=str,
                            help='tb tag')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
        }
