import argparse
import torch


# parse train options
def _get_parser():
    """
    Creates an argument parser for training options.

    Returns:
        argparse.ArgumentParser: The parser containing training options.
    """
    parser = argparse.ArgumentParser(description="Training Configuration Parser")

    # General arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Name of the experiment for tracking or resuming training",
    )
    parser.add_argument("--nr-runs", type=int, default=1, help="Number of training runs")
    parser.add_argument("--seed", type=int, default=7, help="Random seed. default 7")

    # Hardware configuration
    parser.add_argument("--device", type=str, default="cuda", help="Device type, either 'cpu' or 'cuda'")
    parser.add_argument("--device-ids", nargs="+", type=int, default=[5], help="IDs of GPU devices to use")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of workers per GPU for data loading",
    )
    # Dataset-related arguments
    parser.add_argument("--dataset", type=str, default="prostate", help="Dataset to be used for training")
    parser.add_argument(
        "--target-class",
        type=str,
        default="i",
        help="Target class for segmentation: i, o, r",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of data to be used for testing",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to be used for validation",
    )
    parser.add_argument("--input-dim-channels", type=int, default=3, help="Number of input image channels")
    parser.add_argument("--input-dim-size", type=int, default=192, help="Height and width of input images")
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Flag to indicate if images should not be resized",
    )
    parser.add_argument("--augmentation", type=str, default="none", help="Type of data augmentation to apply")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for the first training stage")
    parser.add_argument("--lr-2", type=float, default=1e-4, help="Learning rate for the second stage")
    parser.add_argument("--approach", type=str, default="seq", help="Training approach to use")
    parser.add_argument("--backbone", type=str, default="segformer", help="Model backbone architecture")
    parser.add_argument(
        "--val-best",
        action="store_true",
        default=True,
        help="Use best validation performance to progress to the next domain",
    )
    # Resuming training
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to the directory where the model checkpoints are stored, such as optici-seq",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Epoch number to resume training from, -1 for the latest checkpoint",
    )

    # Loss configuration
    parser.add_argument("--loss-type", type=str, default="dice_bce", help="Type of loss function to use for training")
    parser.add_argument("--lambda-d", type=float, default=0.001, help="Weight for tuning the sub-loss")

    # hyperparameters
    parser.add_argument("--multiply-probes", type=float, default=1, help="n batch of probes")
    parser.add_argument("--boundary", type=float, default=0.99, help="Weight for boundary")
    parser.add_argument("--mask-ratio", type=float, default=0.1, help="Mask ratio for the masks")
    return parser


def parse_args(argv):
    """
    Parses command-line arguments.

    Args:
        argv (list): List of arguments passed from the command line.

    Returns:
        Namespace: Parsed arguments as an object with attribute-style access.
    """
    parser = _get_parser()
    args = parser.parse_args(argv)

    # Determine device based on availability
    args.device = str(
        args.device + ":" + str(args.device_ids[0]) if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    device_name = str(torch.cuda.get_device_name(args.device) if args.device == "cuda" else args.device)
    print(f"Device name: {device_name}")
    args.input_shape = (args.input_dim_channels, args.input_dim_size, args.input_dim_size)

    return args


def parse_args_as_dict(argv):
    """
    Parses command-line arguments and returns them as a dictionary.

    Args:
        argv (list): List of arguments passed from the command line.

    Returns:
        dict: Arguments represented as a dictionary.
    """
    return vars(parse_args(argv))


def parse_dict_as_args(dictionary):
    """
    Converts a dictionary of arguments into a command-line style argument list.

    Args:
        dictionary (dict): Dictionary of argument names and values.

    Returns:
        Namespace: Parsed arguments as an object with attribute-style access.
    """
    argv = []
    for key, value in dictionary.items():
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
        else:
            argv.append(f"--{key}")
            argv.append(str(value))
    return parse_args(argv)
