from models.SWAD import SWAD
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.SWAD.utils import AveragedModel, update_bn, LossValley

class resamplingSWAD(SWAD):
    def __init__(self, opt, wandb):
        super(resamplingSWAD, self).__init__(opt, wandb)
        self.annealing_epochs = opt['swa_annealing_epochs']
        
        self.set_optimizer(opt)
        self.swad = LossValley(n_converge = opt['swad_n_converge'], n_tolerance = opt['swad_n_converge'] + opt['swad_n_tolerance'], 
                               tolerance_ratio = opt['swad_tolerance_ratio'])
        
        self.step = 0
    