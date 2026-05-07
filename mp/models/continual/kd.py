from mp.models.model import Model
import torch.optim as optim


class KD(Model):

    def __init__(
        self,
        input_shape=(1, 256, 256),
        nr_labels=2,
    ):
        r"""Constructor

        Args:
            input_shape (tuple of int): input shape of the images
            nr_labels (int): number of labels for the segmentation
            unet_dropout (float): dropout probability for the U-Net
            unet_monte_carlo_dropout (float): monte carlo dropout probability for the U-Net
            unet_preactivation (boolean): whether to use U-Net pre-activations
        """
        super(KD, self).__init__()

        self.input_shape = input_shape
        self.nr_labels = nr_labels

        self.init_backbone()
