/* p3_ej4b.c - falsa comparticion y el escalon de la linea de cache (apartado 3.3.12).
   Cada hilo incrementa SU casilla; lo unico que cambia es la distancia entre ellas. */
#include <stdio.h>
#include <cpuid.h>
#include <omp.h>
#include "crono.h"

#define VUELTAS 200000000   /* incrementos en total, repartidos entre los hilos */
#define MAXH 64
#define SEPMAX 16
#define REP 5

static volatile double sitio[MAXH * SEPMAX];

/* Tamano de la linea de cache que declara el procesador: instruccion CPUID. */
static int linea_cache(void)
{
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    return (int)((b >> 8) & 0xff) * 8;
}

/* Cada hilo suma uno VUELTAS/p veces en sitio[yo * sep]. */
static double mide(int p, int sep, double *total)
{
    for (int i = 0; i < MAXH * SEPMAX; i++) sitio[i] = 0.0;
    double t0 = reloj();
    #pragma omp parallel num_threads(p)
    {
        int yo = omp_get_thread_num();
        for (int k = 0; k < VUELTAS / p; k++)
            sitio[yo * sep] += 1.0;
    }
    double dt = reloj() - t0;
    double s = 0.0;
    for (int i = 0; i < MAXH * SEPMAX; i++) s += sitio[i];
    *total = s;
    return dt;
}

/* El mismo recuento sin array ninguno: cada hilo en su copia, y reduction.
   OJO: la variable NO es volatile, para que la copia privada viva en un registro. */
static double con_reduction(int p, double *total)
{
    double suma = 0.0;
    double t0 = reloj();
    #pragma omp parallel num_threads(p) reduction(+:suma)
    {
        for (int k = 0; k < VUELTAS / p; k++)
            suma += 1.0;
    }
    double dt = reloj() - t0;
    *total = suma;
    return dt;
}

/* Mejor de REP de mide(p, sep); sep = 0 significa con_reduction. */
static double mejor(int p, int sep, double *total)
{
    double m = 1e30;
    for (int r = 0; r < REP; r++) {
        double t = sep ? mide(p, sep, total) : con_reduction(p, total);
        if (t < m) m = t;
    }
    return m;
}

int main(void)
{
    printf("linea de cache segun CPUID: %d bytes; un double ocupa %d\n", linea_cache(), (int)sizeof(double));
    int hilos[] = {2, 4, 8, 24};
    double tp, ts, tr, sp, ss, sr;
    printf(" hilos pegadas(s) separadas(s) factor reduction(s) sumas\n");
    for (int k = 0; k < 4; k++) {
        int p = hilos[k];
        tp = mejor(p, 1, &sp);
        ts = mejor(p, SEPMAX, &ss);
        tr = mejor(p, 0, &sr);
        double esperado = (double)(VUELTAS / p) * p;
        printf("%6d %11.4f %13.4f %6.1fx %13.4f %s\n", p, tp, ts, tp / ts, tr,
            (sp == esperado && ss == esperado && sr == esperado) ? "bien" : "MAL");
    }
    printf("barrido de la separacion con 8 hilos\n separacion bytes tiempo(s)\n");
    for (int sep = 1; sep <= SEPMAX; sep++)
        printf("%11d %6d %10.4f\n", sep, sep * (int)sizeof(double), mejor(8, sep, &sp));
    return 0;
}
