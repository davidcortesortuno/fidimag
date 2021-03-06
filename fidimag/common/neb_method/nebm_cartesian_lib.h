#include "math.h"
#define WIDE_PI 3.1415926535897932384626433832795L

double compute_distance_cartesian(double * A, double * B, int n_dofs_image,
                                  int * material, int n_dofs_image_material
                                  );

void compute_dYdt_C(double * y, double * G, double * dYdt,
                    int * pins,
                    int n_images, int n_dofs_image);


void project_images_C(double * vector, double * y,
                      int n_images, int n_dofs_image
                      );
