import pyopencl as cl
import numpy as np
import pygame
import time

windows_size = [800, 600]
fullscreen = False

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode(windows_size)
font = pygame.font.SysFont('Arial', 20)

#pygame.display.toggle_fullscreen()

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]

ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, windows_size[0] * windows_size[1] * np.dtype(np.uint16).itemsize)

kernel_code = """
__kernel void mandelbrot(const int width, const int height,
                            const double xmin, const double xmax,
                            const double ymin, const double ymax,
                            __global ushort *output,
                            const ushort maxiter)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    double cx = xmin + (xmax - xmin) * x / width;
    double cy = ymin + (ymax - ymin) * y / height;
    double zx = 0.0f;
    double zy = 0.0f;
    ushort iter = 0;

    while (zx*zx + zy*zy <= 4.0f && iter < maxiter) {
        double xtemp = zx*zx - zy*zy + cx;
        zy = 2*zx*zy + cy;
        zx = xtemp;
        iter++;
    }
    output[y * width + x] = iter;
}
"""
prg = cl.Program(ctx, kernel_code).build()

def mandelbrot_gpu(width, height, maxiter, real_range, imag_range):
    global_work_size = (width, height)
    prg.mandelbrot(queue, global_work_size, None, 
                   np.int32(width), np.int32(height),
                   np.float64(real_range[0]), np.float64(real_range[1]), 
                   np.float64(imag_range[0]), np.float64(imag_range[1]), 
                   output_buf, np.uint16(maxiter))

    output = np.empty((height, width), dtype=np.uint16)
    cl.enqueue_copy(queue, output, output_buf)

    return output

import numpy as np

def zoom_and_pan(real_range, imag_range, zoom_factor, offset):
  pan_x, pan_y = offset
  center_x = (real_range[1] + real_range[0]) / 2
  center_y = (imag_range[1] + imag_range[0]) / 2

  width = (real_range[1] - real_range[0]) * zoom_factor * aspect_ratio
  height = (imag_range[1] - imag_range[0]) * zoom_factor
  
  new_center_x = center_x + pan_x
  new_center_y = center_y + pan_y

  new_real_range = np.array([new_center_x - width/2, new_center_x + width/2], dtype=np.float64)
  new_imag_range = np.array([new_center_y - height/2, new_center_y + height/2], dtype=np.float64)

  return new_real_range, new_imag_range

maxiter_index = 1
maxiter = 255*maxiter_index
aspect_ratio = windows_size[0] / windows_size[1]

zoom = 1.0
offset = [0.0, 0.0]

real_range = np.array([-2, 1], dtype=np.float64)
imag_range = np.array([-1.5, 1.5], dtype=np.float64)

dt = 1/60
running = True
while running:
    debut = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                maxiter -= 255
                maxiter = max(maxiter, 255)
            if event.key == pygame.K_q:
                maxiter += 255
            
            if event.key == pygame.K_f:
                if fullscreen:
                    windows_size = [800, 600]
                    aspect_ratio = windows_size[0] / windows_size[1]
                    pygame.display.toggle_fullscreen()
                    screen = pygame.display.set_mode(windows_size)
                    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, windows_size[0] * windows_size[1] * np.dtype(np.uint16).itemsize)
                    fullscreen = False
                else:
                    windows_size = list(pygame.display.get_desktop_sizes()[0])
                    aspect_ratio = windows_size[0] / windows_size[1]
                    screen = pygame.display.set_mode(windows_size)
                    pygame.display.toggle_fullscreen()
                    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, windows_size[0] * windows_size[1] * np.dtype(np.uint16).itemsize)
                    fullscreen = True

    keys=pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        offset[0] += (zoom*1)*dt
    if keys[pygame.K_LEFT]:
        offset[0] -= (zoom*1)*dt
    if keys[pygame.K_UP]:
        offset[1] -= (zoom*1)*dt
    if keys[pygame.K_DOWN]:
        offset[1] += (zoom*1)*dt
    if keys[pygame.K_z]:
        zoom += (zoom*1)*dt
    if keys[pygame.K_a]:
        zoom -= (zoom*1)*dt
    
    real_range = np.array([-2, 1], dtype=np.float64)
    imag_range = np.array([-1.5, 1.5], dtype=np.float64)

    real_range, imag_range = zoom_and_pan(real_range, imag_range, zoom, offset)

    output = mandelbrot_gpu(windows_size[0], windows_size[1], maxiter, real_range, imag_range)

    surface = pygame.surfarray.make_surface(np.float64(output.swapaxes(0, 1) * 255.0 / maxiter))

    screen.fill((0, 0, 0))
    screen.blit(surface, (0, 0))
    
    text_surface = font.render("fps: "+str(round(1/dt)), True, (255, 255, 255))
    screen.blit(text_surface, (0, 0))
    text_surface = font.render("zoom: "+str(round(1/zoom, 1)), True, (255, 255, 255))
    screen.blit(text_surface, (0, 23))
    text_surface = font.render("maxiter: "+str(maxiter), True, (255, 255, 255))
    screen.blit(text_surface, (0, 23*2))
    
    pygame.display.flip()

    dt = time.time() - debut
    while dt < 1/100:
        dt = time.time() - debut