
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image_write.h"

#define PI 3.14159265358979323846

/*Fonction pour calculer la valeur du noyau gaussien
    Permet de calculer la valeur du noyau gaussien,
    plus la valeur de l'entrée `sigma` sera élevé et plus l'image sera flouter
    plus cette valeur est élevée, plus le temps de calcul sera long
*/
float gaussian(float x, float sigma)
{
    return (1.0 / (2 * PI * sigma * sigma)) * exp(-(x * x) / (2 * sigma * sigma));
}

/* Fonction de calcul des nouvelles valeurs des pixel en appliquant le filtre gaussien
    Paramètres : paramètres de l'image (largeur, hauteur, nombre de canaux de couleur) + sigma pour l'écart type du noyau gaussien
*/
void applyGaussianBlur(const uint8_t *inputPixels, uint8_t *outputPixels, int width, int height, int channels, float sigma)
{
    int radius = (int)(sigma * 3);
    int size = 2 * radius + 1;
    float kernel[size];

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

    // Appliquer le filtre de convolution
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
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

    float sigma = 10.0; // L'écart type du noyau gaussien

    // Appliquer le flou gaussien
    applyGaussianBlur(inputPixels, outputPixels, width, height, channels, sigma);

    // Enregistrer l'image floutée
    stbi_write_bmp("output_cpu.bmp", width, height, channels, outputPixels);

    // Libérer la mémoire
    stbi_image_free(inputPixels);
    free(outputPixels);

    printf("Image floutée enregistrée sous : output_image.bmp\n");

    return 0;
}