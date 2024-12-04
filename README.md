# pyopencl_mandelbrot_fractal

**This Python project leverages the power of OpenCL to render the Mandelbrot set fractal in real-time. By offloading the computationally intensive task to the GPU, it achieves significant performance gains compared to pure CPU-based implementations.**

## Getting started
```
git clone https://github.com/legoman0701/pyopencl_mandelbrot_fractal.git
python -m pip install pyopencl numpy pygame
```

## Usage

```
python main.py
```

## Controls
```
F: Toggle fullscreen mode
Arrow keys: Pan the view
A/Z: Zoom in/out
Q/S: Adjust the maximum iteration count (influences detail and color)
```

## How it Works

Initializes an OpenCL context, connecting to the available GPU devices.
Compiles the OpenCL kernel code, which defines the parallel computation for each pixel.
Allocates device memory for input parameters (image dimensions, iteration count, etc.) and output buffer (the rendered image).
Enqueues the kernel to the command queue, specifying the work group size and global work size.
Transfers the rendered image data from device memory back to the host.
Renders the image using a pygame.

## Images
![main](https://github.com/legoman0701/pyopencl_mandelbrot_fractal/blob/main/images/Capture%20d’écran%20(2).png)
![1](https://github.com/legoman0701/pyopencl_mandelbrot_fractal/blob/main/images/Capture%20d’écran%20(3).png)
![2](https://github.com/legoman0701/pyopencl_mandelbrot_fractal/blob/main/images/Capture%20d’écran%20(4).png)
