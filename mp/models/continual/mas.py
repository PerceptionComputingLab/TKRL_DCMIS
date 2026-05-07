from mp.models.model import Model


class MAS(Model):

    def __init__(
        self,
        input_shape=(1, 256, 256),
        nr_labels=2,
    ):
        r"""Constructor

        Args:
            input_shape (tuple of int): input shape of the images
            nr_labels (int): number of labels for the segmentation
        """
        super(MAS, self).__init__()

        self.input_shape = input_shape
        self.nr_labels = nr_labels

        self.init_backbone()

        self.importance_weights = None
        self.tasks = 0

        self.n_params_backbone = sum(p.numel() for p in self.backbone_new.parameters())

    def update_importance_weights(self, importance_weights):
        r"""Update importance weights w/ computed ones

        Args:
            (torch.Tensor or list): importance_weights
        """
        if self.importance_weights == None:
            self.importance_weights = importance_weights
        else:
            for i in range(len(self.importance_weights)):
                self.importance_weights[i] -= self.importance_weights[i] / self.tasks
                self.importance_weights[i] += importance_weights[i] / self.tasks
        self.tasks += 1
