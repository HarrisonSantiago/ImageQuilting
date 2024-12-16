import torch
import argparse
import textureSynthesis
import textureTransfer
import sys
from torchvision import io
import os


# Get parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--synthesis", action="store_true",
                    help="perform synthesis")
parser.add_argument("--transfer", action="store_true",
                    help="perform transfer")
parser.add_argument('--output_path', type=str, default='output.png',
                    help='output path for generated image')
parser.add_argument("-i1", "--texture_img_path", type=str,
                    help="path of texture image")
parser.add_argument("-i2", "--target_img_path", type=str,
                    help="path of target image")
parser.add_argument("-b", "--block_size", type=int, default=100,
                    help="block size in pixels")
parser.add_argument("-o", "--overlap", type=int, default=20,
                    help="overlap size in pixels")
parser.add_argument("-s", "--scale", type=float, default=2,
                    help="final shape of synthesis image")
parser.add_argument("-m", "--mode", type=str, default='cut',
                    help="synthesis mode to use")
parser.add_argument('-a', '--alpha', type=float, default=0.1,
                    help="alpha value for transfer")
parser.add_argument('-l', '--level', type=int, default=0,
                    help='iteration level for hierarchical synthesis')
parser.add_argument('-p', '--prior', type=torch.Tensor, default=None,
                    help='previous iteration')
parser.add_argument('-g', '--gaussian_blur', type=bool, default=False,
                    help='gaussian blur for correlation')
parser.add_argument('-n', '--n_steps', type=int, default=-1,
                    help='num steps for transfer iter')
parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for computation')
parser.add_argument('-al', '--algorithm', type=str, default='dijkstra',
                    help='path algorithm')


args = parser.parse_args()


def synthesis(args):
    """
    Wrapper function for texture synthesis.

    Args:
        args: Argument namespace containing:
            texture_img_path (str): Path to input texture image
            scale (float): Factor to scale the output image
            block_size (int): Size of patches used in synthesis
            overlap (int): Overlap between patches
            mode (str): Synthesis mode ("random", "best", or "cut")
            device (str): Device to use for computation
            output_path (str, optional): Path for output image.
                                         Defaults to "output.png"

    Returns:
        Tensor: Synthesized image tensor

    Raises:
        FileNotFoundError: If input texture file doesn't exist
        ValueError: If scale, block_size, or overlap are invalid
    """
    # Input validation
    if not os.path.exists(args.texture_img_path):
        raise FileNotFoundError("Texture image not found")

    if args.scale <= 0:
        raise ValueError(f"Scale must be positive, got {args.scale}")

    if args.block_size <= 0:
        raise ValueError(f"Block size must be positive, got {args.block_size}")

    if args.overlap < 0 or args.overlap >= args.block_size:
        raise ValueError("Overlap must be between 0 and block_size, " +
                         f"got {args.overlap}")

    texture = io.decode_image(args.texture_img_path).float() / 255.0
    texture = texture.to(args.device)
    print(texture.shape)
    texture = texture.permute(1, 2, 0)

    # Calculate output dimensions
    h, w = texture.shape[:2]
    new_h, new_w = int(h * args.scale), int(w * args.scale)
    print(f'{new_h=}')
    print(f'{new_w=}')

    # Perform synthesis
    try:
        img = textureSynthesis.synthesize(
            texture=texture,
            patchLength=args.block_size,
            overlap=args.overlap,
            shape=[new_h, new_w],
            mode=args.mode,
            algorithm=args.algorithm,
            device=args.device
        )

        img = (img.clamp(0, 1) * 255).to(torch.uint8)
        img = img.permute(2,0,1)

        output_path = getattr(args, 'output_path', 'synthesis.png')
        io.write_png(img.cpu(), output_path)

        return img

    except Exception as e:
        raise RuntimeError(f"Synthesis failed: {e}")


def transfer(args):
    """
    Wrapper function for texture transfer.

    Args:
        args: Argument namespace containing:
            texture_img_path (str): Path to source texture image
            target_img_path (str): Path to target image
            n_steps (int): Number of hierarchical refinement steps
            block_size (int): Size of patches used in transfer
            mode (str, optional): Transfer mode ("best", "overlap", or "cut")
            overlap (int, optional): Overlap between patches
            alpha (float, optional): Weight between structure and texture (0 to 1)
            level (int, optional): Current hierarchical level
            prior (Tensor, optional): Result from previous iteration
            gaussian_blur (bool, optional): Whether to blur correlation images
            device (str): Device to use for computation
            output_path (str, optional): Path for output image

    Returns:
        Tensor: Transferred image tensor

    Raises:
        FileNotFoundError: If input images don't exist
        ValueError: If numeric parameters are invalid
    """
    # Input validation
    if not os.path.exists(args.texture_img_path):
        raise FileNotFoundError("Texture image not found")

    if not os.path.exists(args.target_img_path):
        raise FileNotFoundError("Target image not found")

    if args.block_size <= 0:
        raise ValueError(f"Block size must be positive, got {args.block_size}")

    if hasattr(args, 'overlap') and \
            (args.overlap < 0 or args.overlap >= args.block_size):
        raise ValueError(f"Overlap must be between 0 and block_size, got {args.overlap}")

    if hasattr(args, 'alpha') and (args.alpha < 0 or args.alpha > 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {args.alpha}")

    # Load images using torchvision
    texture_img = io.read_image(args.texture_img_path)[:3].float() / 255.0  # Keep only RGB
    target_img = io.read_image(args.target_img_path)[:3].float() / 255.0

    texture_img = texture_img.to(args.device)
    target_img = target_img.to(args.device)

    texture_img = texture_img.permute(1,2,0)
    target_img = target_img.permute(1,2,0)

    # Check image compatibility
    if texture_img.dim() != target_img.dim():
        raise ValueError("Texture and target images must have same number of dimensions")

    # Perform transfer
    try:
        if args.n_steps > 0:
            img = textureTransfer.transfer_iter(
                texture=texture_img,
                target=target_img,
                patchLength=args.block_size,
                n=args.n_steps,
                device=args.device
            )
        else:
            img = textureTransfer.transfer(
                texture=texture_img,
                target=target_img,
                patchLength=args.block_size,
                overlap=args.overlap,
                mode=args.mode,
                alpha=args.alpha,
                level=args.level,
                prior=args.prior.to(args.device) if args.prior is not None else None,
                blur=args.gaussian_blur,
                device=args.device
            )

        img = (img.clamp(0, 1) * 255).to(torch.uint8)
        img = img.permute(2,0,1)
        output_path = getattr(args, 'output_path', 'transfer.png')
        io.write_png(img.cpu(), output_path)

        return img

    except Exception as e:
        raise RuntimeError(f"Transfer failed: {e}")


if __name__ == "__main__":
    if (args.synthesis and args.transfer):
        print("Cannot perform synthesis & transfer simultaneously")
        sys.exit(1)
    elif args.synthesis:
        synthesis(args)
    elif args.transfer:
        transfer(args)
