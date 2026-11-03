
#include <stdio.h>
#include "crono.h"

int main(void)
{
#ifndef _OPENMP
    printf("ALGO FALLA: se ha compilado sin -fopenmp\n");
    return 1;
#else
    int en_region = 0;

    #pragma omp parallel reduction(+:en_region)
    en_region += 1;

    double paso = 1.0;

    for (int i = 0; i < 1000; i++) {
        double a = reloj(), b;
        do b = reloj(); while (b == a);
        if (b - a < paso) paso = b - a;
    }

    int maximo = omp_get_max_threads();

    printf("procesadores logicos (omp_get_num_procs) : %d\n", omp_get_num_procs());
    printf("hilos por defecto (omp_get_max_threads) : %d\n", maximo);
    printf("hilos que han ejecutado la region         : %d\n", en_region);
    printf("resolucion del reloj de crono.h           : %.0f ns\n", paso * 1e9);

    printf("%s\n", (en_region > 1 && en_region == maximo && paso < 1e-6)
           ? "ENTORNO LISTO" : "ALGO FALLA: avisa antes de la sesion");

    return 0;
#endif
}
