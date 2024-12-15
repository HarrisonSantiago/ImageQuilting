import numpy as np
import argparse
import textureSynthesis
import textureTransfer
import sys
from skimage import io
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
                    help='iteration level for hierarchical synthesis, recommend 0')
parser.add_argument('-p', '--prior', type=np.ndarray, default=None,
                    help='previous iteration')
parser.add_argument('-g', '--gaussian_blur', type=bool, default=False,
                    help='gaussian blur for correlation')
parser.add_argument('-n', '--n_steps', type=int, default=-1,
                    help='num steps for transfer iter')


args = parser.parse_args()


def synthesis(args):
    """
    Wrapper function for texture synthesis that handles image loading, scaling, and saving.
    
    Args:
        args: Argument namespace containing:
            texture_img_path (str): Path to input texture image
            scale (float): Factor to scale the output image
            block_size (int): Size of patches used in synthesis
            overlap (int): Overlap between patches
            mode (str): Synthesis mode ("random", "best", or "cut")
            output_path (str, optional): Path for output image. Defaults to "output.png"
    
    Returns:
        ndarray: Synthesized image array
        
    Raises:
        FileNotFoundError: If input texture file doesn't exist
        ValueError: If scale, block_size, or overlap are invalid
    """
    # Input validation
    if not os.path.exists(args.texture_img_path):
        raise FileNotFoundError(f"Texture image not found: {args.texture_img_path}")
    
    if args.scale <= 0:
        raise ValueError(f"Scale must be positive, got {args.scale}")
        
    if args.block_size <= 0:
        raise ValueError(f"Block size must be positive, got {args.block_size}")
        
    if args.overlap < 0 or args.overlap >= args.block_size:
        raise ValueError(f"Overlap must be between 0 and block_size, got {args.overlap}")
    
    texture = io.imread(args.texture_img_path)
        
    # Calculate output dimensions
    h, w = texture.shape[:2]
    new_h, new_w = int(h * args.scale), int(w * args.scale)
    
    # Perform synthesis
    try:
        img = textureSynthesis.synthesize(
            texture=texture,
            patchLength=args.block_size,
            overlap=args.overlap,
            shape=[new_h, new_w],
            mode=args.mode
        )
        
        # Convert to 8-bit image
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Save result
        output_path = getattr(args, 'output_path', 'synthesis.png')
        io.imsave(output_path, img)
        
        return img
        
    except Exception as e:
        raise RuntimeError(f"Synthesis failed: {e}")
    

def transfer(args):
    """
    Wrapper function for texture transfer that handles image loading, processing, and saving.
    
    Args:
        args: Argument namespace containing:
            texture_img_path (str): Path to source texture image
            target_img_path (str): Path to target image
            n_steps (int): Number of hierarchical refinement steps (if > 0, uses transfer_iter)
            block_size (int): Size of patches used in transfer
            mode (str, optional): Transfer mode ("best", "overlap", or "cut")
            overlap (int, optional): Overlap between patches
            alpha (float, optional): Weight between structure and texture (0 to 1)
            level (int, optional): Current hierarchical level
            prior (ndarray, optional): Result from previous iteration
            gaussian_blur (bool, optional): Whether to blur correlation images
            output_path (str, optional): Path for output image. Defaults to "transfer.png"
    
    Returns:
        ndarray: Transferred image array
        
    Raises:
        FileNotFoundError: If input images don't exist
        ValueError: If numeric parameters are invalid
    """
    # Input validation
    if not os.path.exists(args.texture_img_path):
        raise FileNotFoundError(f"Texture image not found: {args.texture_img_path}")
    
    if not os.path.exists(args.target_img_path):
        raise FileNotFoundError(f"Target image not found: {args.target_img_path}")
        
    if args.block_size <= 0:
        raise ValueError(f"Block size must be positive, got {args.block_size}")
        
    if hasattr(args, 'overlap') and (args.overlap < 0 or args.overlap >= args.block_size):
        raise ValueError(f"Overlap must be between 0 and block_size, got {args.overlap}")
        
    if hasattr(args, 'alpha') and (args.alpha < 0 or args.alpha > 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {args.alpha}")
        
    texture_img = io.imread(args.texture_img_path)[:, :, :3]  #RGBA or RGB
    target_img = io.imread(args.target_img_path)[:, :, :3]

    print(texture_img.shape)
    print(target_img.shape)

        
    # Check image compatibility
    if texture_img.ndim != target_img.ndim:
        raise ValueError("Texture and target images must have same number of channels")
    
    # Perform transfer
    try:
        if args.n_steps > 0:
            img = textureTransfer.transfer_iter(
                texture=texture_img,
                target=target_img,
                patchLength=args.block_size,
                n=args.n_steps
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
                prior=args.prior,
                blur=args.gaussian_blur
            )
            
        # Convert to 8-bit image
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Save result
        output_path = getattr(args, 'output_path', 'transfer.png')
        io.imsave(output_path, img)
        
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
