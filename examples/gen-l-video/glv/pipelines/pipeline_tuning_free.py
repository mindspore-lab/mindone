class TuningFreePipeline:
    def __init__(self, sd, unet, scheduler, depth_estimator):
        super().__init__()

        self.sd = sd
        self.unet = unet
        self.scheduler = scheduler
        self.depth_estimator = depth_estimator
