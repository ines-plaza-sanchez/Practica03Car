
/* p3_ej1.c - cuales se pueden paralelizar y cuales no */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define N    2000000
#define EJEC 10

static double a[N], b[N], ref[N], copia[N];

static void inicial(double *v)
{
    for (int i = 0; i < N; i++)
        v[i] = (double)(i % 7) + 1.0;
}

/* Posiciones en las que v no coincide con la referencia. */
static long long difiere(const double *v)
{
    long long d = 0;

    for (int i = 0; i < N; i++)
        if (v[i] != ref[i]) d++;

    return d;
}

static void informa(const char *nombre, int malas, long long peor)
{
    printf("%-34s %4d/%d %14lld\n", nombre, malas, EJEC, peor);
}

int main(void)
{
    printf("%d hilos, %d posiciones, %d ejecuciones por bucle\n",
           omp_get_max_threads(), N, EJEC);
    printf("bucle                              fallos posiciones mal (peor)\n");

    /* 1) Sin dependencias: sirve de control */
    inicial(ref);

    for (int i = 0; i < N; i++)
        ref[i] = 2.0 * ref[i];

    {
        int malas = 0;
        long long peor = 0;

        for (int e = 0; e < EJEC; e++) {
            inicial(b);

            #pragma omp parallel for
            for (int i = 0; i < N; i++)
                b[i] = 2.0 * b[i];

            long long d = difiere(b);

            if (d) {
                malas++;
                if (d > peor) peor = d;
            }
        }

        informa("1 b[i] = 2*b[i]", malas, peor);
    }

    /* 2) Dependencia de flujo: cada vuelta necesita la anterior */
    inicial(a);
    ref[0] = a[0];

    for (int i = 1; i < N; i++)
        ref[i] = ref[i - 1] + a[i];

    {
        int malas = 0;
        long long peor = 0;

        for (int e = 0; e < EJEC; e++) {
            inicial(b);

            #pragma omp parallel for
            for (int i = 1; i < N; i++)
                b[i] = b[i - 1] + b[i];

            long long d = difiere(b);

            if (d) {
                malas++;
                if (d > peor) peor = d;
            }
        }

        informa("2 b[i] = b[i-1] + b[i]", malas, peor);
    }

    /* 3) Antidependencia: se lee una posición que otro hilo puede escribir */
    inicial(ref);

    for (int i = 0; i < N - 1; i++)
        ref[i] = ref[i + 1] + 1.0;

    {
        int malas = 0;
        long long peor = 0;

        for (int e = 0; e < EJEC; e++) {
            inicial(b);

            #pragma omp parallel for
            for (int i = 0; i < N - 1; i++)
                b[i] = b[i + 1] + 1.0;

            long long d = difiere(b);

            if (d) {
                malas++;
                if (d > peor) peor = d;
            }
        }

        informa("3 b[i] = b[i+1] + 1", malas, peor);
    }

    /* 3 arreglado: se utiliza una copia del array original */
    {
        int malas = 0;
        long long peor = 0;

        for (int e = 0; e < EJEC; e++) {
            inicial(b);

            /* Guardamos los valores originales antes de paralelizar */
            memcpy(copia, b, sizeof(b));

            #pragma omp parallel for
            for (int i = 0; i < N - 1; i++)
                b[i] = copia[i + 1] + 1.0;

            long long d = difiere(b);

            if (d) {
                malas++;
                if (d > peor) peor = d;
            }
        }

        informa("3 arreglado", malas, peor);
    }

    /* 4) y 5) Un acumulador compartido, sin y con reduction */
    inicial(a);

    double s_ref = 0.0;

    for (int i = 0; i < N; i++)
        s_ref += a[i];

    {
        int malas = 0;
        double peor = 0.0;

        for (int e = 0; e < EJEC; e++) {
            double s = 0.0;

            #pragma omp parallel for
            for (int i = 0; i < N; i++)
                s += a[i];

            double err = fabs(s - s_ref) / s_ref;

            if (err > 0.0) {
                malas++;
                if (err > peor) peor = err;
            }
        }

        printf("%-34s %4d/%d error relativo %.3f\n",
               "4 s += a[i] (sin reduction)", malas, EJEC, peor);
    }

    {
        int malas = 0;

        for (int e = 0; e < EJEC; e++) {
            double s = 0.0;

            #pragma omp parallel for reduction(+:s)
            for (int i = 0; i < N; i++)
                s += a[i];

            if (s != s_ref) malas++;
        }

        printf("%-34s %4d/%d\n",
               "5 s += a[i] (con reduction)", malas, EJEC);
    }

    return 0;
}
