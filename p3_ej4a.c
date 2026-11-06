/* p3_ej4a.c - el temporal que creias privado (apartado 3.3.3).
Cada version se ejecuta EJEC veces y se compara con la secuencial posicion a posicion. Compilalo con -O2 y con -O0 y compara las dos salidas. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 4000000
#define EJEC 20

static double *x, *y, *ref;

/* Posiciones en las que y no coincide con la referencia secuencial. */
static long long difiere(void)
{
    long long d = 0;
    for (int i = 0; i < N; i++)
        if (y[i] != ref[i]) d++;
    return d;
}

static void informa(const char *nombre, int malas, long long peor)
{
    printf("%-38s %5d/%d %14lld\n", nombre, malas, EJEC, peor);
}

/* A: la variable auxiliar se declara fuera de la region */
static void version_a(void)
{
    int malas = 0; long long peor = 0;
    for (int e = 0; e < EJEC; e++) {
        double t;                       /* fuera de la region: COMPARTIDA */
        #pragma omp parallel for shared(t)
        for (int i = 0; i < N; i++) {
            t = sqrt(x[i]) + 1.0;
            y[i] = t * t;
        }
        long long d = difiere();
        if (d) { malas++; if (d > peor) peor = d; }
    }
    informa("A temporal compartida", malas, peor);
}

/* A2: el mismo error, con la variable obligada a vivir en memoria */
static void version_a2(void)
{
    int malas = 0; long long peor = 0;
    for (int e = 0; e < EJEC; e++) {
        volatile double t;              /* volatile: nunca en un registro */
        #pragma omp parallel for shared(t)
        for (int i = 0; i < N; i++) {
            t = sqrt(x[i]) + 1.0;
            y[i] = t * t;
        }
        long long d = difiere();
        if (d) { malas++; if (d > peor) peor = d; }
    }
    informa("A2 la misma, con t volatile", malas, peor);
}

/* B: la misma variable de fuera, con private(t).
   Cada hilo recibe su propia copia de t. */
static void version_b(void)
{
    int malas = 0; long long peor = 0;
    for (int e = 0; e < EJEC; e++) {
        double t;                       /* fuera de la region, pero PRIVADA */
        #pragma omp parallel for private(t)
        for (int i = 0; i < N; i++) {
            t = sqrt(x[i]) + 1.0;
            y[i] = t * t;
        }
        long long d = difiere();
        if (d) { malas++; if (d > peor) peor = d; }
    }
    informa("B private(t)", malas, peor);
}

/* C: la variable declarada dentro del bucle, y la region con default(none).
   Lo declarado dentro de la region es privado de cada hilo por construccion. */
static void version_c(void)
{
    int malas = 0; long long peor = 0;
    for (int e = 0; e < EJEC; e++) {
        #pragma omp parallel for default(none) shared(x, y)
        for (int i = 0; i < N; i++) {
            double t = sqrt(x[i]) + 1.0;   /* dentro: privada */
            y[i] = t * t;
        }
        long long d = difiere();
        if (d) { malas++; if (d > peor) peor = d; }
    }
    informa("C declarada dentro + default(none)", malas, peor);
}

int main(void)
{
    x = malloc(N * sizeof *x);
    y = malloc(N * sizeof *y);
    ref = malloc(N * sizeof *ref);
    if (!x || !y || !ref) { fprintf(stderr, "sin memoria\n"); return 1; }
    for (int i = 0; i < N; i++) {
        x[i] = 1.0 + (double)(i % 1000);
        double t = sqrt(x[i]) + 1.0;    /* la version secuencial */
        ref[i] = t * t;
    }
    printf("%d hilos, %d posiciones, %d ejecuciones por version\n", omp_get_max_threads(), N, EJEC);
    printf("version ejecuciones mal posiciones mal\n");
    version_a();
    version_a2();
    version_b();
    version_c();
    free(x);
    free(y);
    free(ref);
    return 0;
}
