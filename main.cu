#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <iostream>

// On importe les fichiers pour utiliser un floutage gaussien
#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image_write.h"

#define PI 3.14159265358979323846

/*Fonction executable sur GPU
    Permet de calculer la valeur du noyau gaussien,
    plus la valeur de l'entrée `sigma` sera élevé et plus l'image sera flouter
    plus cette valeur est élevée, plus le temps de calcul sera long
*/
__device__ float gaussian(float x, float sigma)
{
    return (1.0 / (2 * PI * sigma * sigma)) * exp(-(x * x) / (2 * sigma * sigma));
}

/* Fonction de calcul des nouvelles valeurs des pixel en appliquant le filtre gaussien
    Paramètres : paramètres de l'image (largeur, hauteur, nombre de canaux de couleur) + sigma pour l'écart type du noyau gaussien
    Les threads calculent la valeur d'un pixel de l'image de sortie après l'application du filtre gaussien.
*/
__global__ void applyGaussianBlur(const uint8_t *inputPixels, uint8_t *outputPixels, int width, int height, int channels, float sigma)
{
    int radius = (int)(sigma * 3);
    int size = 2 * radius + 1;

    float kernel[61];

    // construction du noyau de convo gaussien
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        kernel[i] = gaussian(i - radius, sigma);
        sum += kernel[i];
    }

    // Nomalisation du noyau
    /*Le masque resemble à :
        { kernel[0] / sum, kernel[0] / sum, kernel[0] / sum },
        { kernel[1] / sum, kernel[1] / sum, kernel[1] / sum },
        { kernel[2] / sum, kernel[2] / sum, kernel[2] / sum }
    */
    for (int i = 0; i < size; i++)
    {
        kernel[i] /= sum;
    }

    // x et y => pixel de l'image
    // associe à chaque thread un pixel donnée
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) // vérifie que les variables x et y ne depasse pas les limites de la taille de l'image
    {
        // Appliquer le noyau de convolution on commence par passer sur chacun des canaux sachant que chaque canal est un tableau de pixels car c est une image RGB
        for (int c = 0; c < channels; c++)
        {
            // Initialiser la nouvelle valeur du pixel
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
    /*Chargement de l'image ("input.bmp") avec allocation mémoire CPU
        +stocke ses pixels dans un tab : "inputPixels"
        +déclaration du tab des nouvelles valeur après filtre : "outpuPixels"
    */
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

    float sigma = 10.0; // L'écart type du noyau gaussien

    uint8_t *d_inputPixels, *d_outputPixels; // init des var alloué sur cuda pour calculs des pixels; variable d'entrée + variable de sortie
    cudaError_t cudaStatus;

    /*Allocation mémoire sur le GPU*/
    cudaStatus = cudaMalloc((void **)&d_inputPixels, width * height * channels * sizeof(uint8_t)); // tab d_inputPixels => pixels de l'image d'entrée (dans cuda)
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaStatus = cudaMalloc((void **)&d_outputPixels, width * height * channels * sizeof(uint8_t)); // tab d_outputPixels => pixels de l'image en sortie (de cuda)
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_inputPixels);
        return 1;
    }

    /*transfert data CPU vers GPU*/
    cudaStatus = cudaMemcpy(d_inputPixels, inputPixels, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_inputPixels);
        cudaFree(d_outputPixels);
        return 1;
    }

    /*Calcul de la configuration des blocs et des grilles*/
    dim3 threadsPerBlock(16, 16);                                                                                              // nb de thread par bloc
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y); // taille des grilles

    // Appliquer le flou gaussien (appel du kernel cuda)
    applyGaussianBlur<<<numBlocks, threadsPerBlock>>>(d_inputPixels, d_outputPixels, width, height, channels, sigma);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_inputPixels);
        cudaFree(d_outputPixels);
        return 1;
    }

    // Copie du résultat du GPU vers le CPU
    cudaMemcpy(outputPixels, d_outputPixels, width * height * channels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // enregistrement de l'image floutée
    stbi_write_bmp("output.bmp", width, height, channels, outputPixels);

    // Libérer la mémoire alloué (CPU et GPU)
    stbi_image_free(inputPixels);
    free(outputPixels);
    cudaFree(d_inputPixels);
    cudaFree(d_outputPixels);

    printf("Image floutée enregistrée sous : output.bmp\n");

    return 0;
}