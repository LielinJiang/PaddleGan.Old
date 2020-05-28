import random


class RandomCrop(object):
    """Crop the input data to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 1.33) of the original aspect ratio is made.
    After applying crop transfrom, the input data will be resized to given size.

    Args:
        output_size (int|list|tuple): Target size of output image, with (height, width) shape.
        scale (list|tuple): Range of size of the origin size cropped. Default: (0.08, 1.0)
        ratio (list|tuple): Range of aspect ratio of the origin aspect ratio cropped. Default: (0.75, 1.33)

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import RandomResizedCrop

            transform = RandomResizedCrop(224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self,
                 output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def _get_params(self, img):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w, _ = img.shape
        th, tw = self.output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        x, y, w, h = self._get_params(img)
        cropped_img = img[y:y + h, x:x + w]
        return cropped_img