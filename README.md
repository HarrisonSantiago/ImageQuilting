# ImageQuilting

This repo provides a Pytorch implementation of the Image Quilting algorithm for texture synthesis and transfer by Efros and Freedman (2001). 

## Instalation
1) `git clone https://github.com/HarrisonSantiago/ImageQuilting.git`
2) `pip install torch`

## Texture Synthesis Usage

Texture synthesis is simple, just name the original texture and the desired image size. Below we show examples of the original texture, generated texture, and the command used. 

# Image Transformation Examples

| Original | Transformed | Command |
|----------|-------------|---------|
| ![Original Image](images\animal_hair.png) | ![Transformed Image](results\animal_hair_b_25_o_10_s_1p5.png) | `python.exe .\main.py --synthesis --output_path .\results\animal_hair_b_25_o_10_s_1p5.png -i1 .\images\animal_hair.png --block_size 50 --overlap 10 --scale 1.5` |
| ![Original Image](images\circles.png) | ![Transformed Image](results\circles_b_50_o_10_s_1p5.png) | `python main.py --synthesis --output_path .\results\circles_b_35_o_5_s_1p5.png -i1 .\images\circles.png -b 35 -o 5 -s 1.5` | 
| ![Original Image](images\wallpaper.jpeg) | ![Transformed Image](results\wallpaper_b_50_o_10_s_1p5.png) | `python main.py --synthesis --output_path .\results\wallpaper_b_50_o_10_s_1p5.png -i1 .\images\wallpaper.jpeg -b 50 -o 10 -s 1.5` | 





## Texture transfer usage

Texture transfer is similarly simple. Name the original texture and the target image. Examples are shown below

# Image Transformation Examples

| Original | Target | Transformed | Command |
|----------|-------------|-------------|---------|
| ![Original Image](path/to/original1.jpg) | ![Transformed Image](path/to/transformed1.jpg) | `./transform --filter blur --radius 5 input.jpg output.jpg` |
| ![Original Image](path/to/original2.jpg) | ![Transformed Image](path/to/transformed2.jpg) | `./transform --grayscale input.jpg output.jpg` |