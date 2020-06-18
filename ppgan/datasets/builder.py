import paddle

from ..utils.registry import Registry


DATASETS = Registry("DATASETS")
        

def build_dataloader(cfg, is_train=True):
    dataset = DATASETS.get(cfg.name)(cfg)
    place = paddle.CUDAPlace(0)
    dataloader = paddle.io.DataLoader(dataset,
                                    batch_size=1,
                                    places=place,
                                    shuffle=True if is_train else False, 
                                    num_workers=0)
    return dataloader