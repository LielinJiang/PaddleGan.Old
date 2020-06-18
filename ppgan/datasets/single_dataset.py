import cv2
import paddle
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset

from .builder import DATASETS


@DATASETS.register()
class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, cfg):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, cfg)
        self.A_paths = sorted(make_dataset(cfg.dataroot, cfg.max_dataset_size))
        input_nc = self.cfg.output_nc if self.cfg.direction == 'BtoA' else self.cfg.input_nc
        self.transform = get_transform(cfg.transform, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        A_img = cv2.imread(A_path)
        A = self.transform(A_img)
        return (A, index) #{'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def get_path_by_indexs(self, indexs):
        if isinstance(indexs, paddle.Variable):
            indexs = indexs.numpy()
        current_paths = []
        for index in indexs:
            current_paths.append(self.A_paths[index])
        return current_paths
