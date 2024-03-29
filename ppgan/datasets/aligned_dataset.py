import cv2
import paddle
import os.path
from .base_dataset import BaseDataset, get_params, get_transform
from .image_folder import make_dataset

from .builder import DATASETS


@DATASETS.register()
class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.cfg.transform.load_size >= self.cfg.transform.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.cfg.output_nc if self.cfg.direction == 'BtoA' else self.cfg.input_nc
        self.output_nc = self.cfg.input_nc if self.cfg.direction == 'BtoA' else self.cfg.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        AB = cv2.imread(AB_path)
        # B_img = cv2.imread(B_path)
        # split AB image into A and B
        h, w = AB.shape[:2]
        # w, h = AB.size
        w2 = int(w / 2)
        # print('type before:', type(AB))
        A = AB[:h, :w2, :]
        B = AB[:h, w2:, :]
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))
        # print('type after:', type(A), type(B), B.SHAPE)

        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        transform_params = get_params(self.cfg.transform, (w2, h))
        # cv2.resize(A, ())
        # a = cv2.resize(B, (286, 286), interpolation=2)
        # print('resize A:', a.shape, type(a), a.dtype)

        A_transform = get_transform(self.cfg.transform, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.cfg.transform, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return A, B, index #{'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def get_path_by_indexs(self, indexs):
        if isinstance(indexs, paddle.Variable):
            indexs = indexs.numpy()
        current_paths = []
        for index in indexs:
            current_paths.append(self.AB_paths[index])
        return current_paths
