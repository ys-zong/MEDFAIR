from models.baseline import baseline


class resampling(baseline):
    def __init__(self, opt, wandb):
        super(resampling, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)
    