#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image_write.h"

#define PI 3.14159265358979323846

__device__ float gaussian(float x, float sigma)
{
    return (1.0 / (2 * PI * sigma * sigma)) * exp(-(x * x) / (2 * sigma * sigma));
}

__global__ void applyGaussianBlur(const uint8_t *inputPixels, uint8_t *outputPixels, int width, int height, int channels, float sigma)
{
    int radius = (int)(sigma * 3);
    int size = 2 * radius + 1;
    float kernel[31];

    // Construire le noyau de convolution gaussien
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        kernel[i] = gaussian(i - radius, sigma);
        sum += kernel[i];
    }

    // Normaliser le noyau
    for (int i = 0; i < size; i++)
    {
        kernel[i] /= sum;
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        for (int c = 0; c < channels; c++)
        {
            float newValue = 0.0;
            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int px = x + kx;
                    int py = y + ky;

                    if (px >= 0 && px < width && py >= 0 && py < height)
                    {
                        newValue += inputPixels[(py * width + px) * channels + c] * kernel[kx + radius] * kernel[ky + radius];
                    }
                }
            }
            outputPixels[(y * width + x) * channels + c] = (uint8_t)newValue;
        }
    }
}

int main()
{
    int width, height, channels;
    uint8_t *inputPixels = stbi_load("./input.bmp", &width, &height, &channels, 0);
    if (!inputPixels)
    {
        printf("Impossible de charger l'image.\n");
        return 1;
    }

    uint8_t *outputPixels = (uint8_t *)malloc(width * height * channels * sizeof(uint8_t));
    if (!outputPixels)
    {
        printf("Erreur lors de l'allocation de mémoire.\n");
        stbi_image_free(inputPixels);
        return 1;
    }

    float sigma = 5.0; // L'écart type du noyau gaussien

    uint8_t *d_inputPixels, *d_outputPixels;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void **)&d_inputPixels, width * height * channels * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaStatus = cudaMalloc((void **)&d_outputPixels, width * height * channels * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_inputPixels);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_inputPixels, inputPixels, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_inputPixels);
        cudaFree(d_outputPixels);
        return 1;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Appliquer le flou gaussien
    applyGaussianBlur<<<numBlocks, threadsPerBlock>>>(d_inputPixels, d_outputPixels, width, height, channels, sigma);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_inputPixels);
        cudaFree(d_outputPixels);
        return 1;
    }

    cudaMemcpy(outputPixels, d_outputPixels, width * height * channels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Print some output pixel values for verification
    std::cout << "Output pixels: ";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << (int)outputPixels[i] << " ";
    }
    std::cout << std::endl;

    // Enregistrer l'image floutée
    stbi_write_bmp("output.bmp", width, height, channels, outputPixels);

    // Libérer la mémoire
    stbi_image_free(inputPixels);
    free(outputPixels);
    cudaFree(d_inputPixels);
    cudaFree(d_outputPixels);

    printf("Image floutée enregistrée sous : output_image.bmp\n");

    return 0;
}
