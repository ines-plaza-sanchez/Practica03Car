/* p3_ej2.c - un bucle desequilibrado y las planificaciones (apartado 3.2.4).
La iteracion i cuesta proporcional a i. */
#include <stdio.h>
#include <omp.h>
#include "crono.h"

#define N 20000
#define REP 5
#define MAXH 64

static double r[N];

/* Trabajo de la iteracion i: sumas en coma flotante. */
static double trabajo (int i)
{
    double s = 0.0;
    for (int k = 0; k < i; k++) s += (double)k * 1e-9;
    return s;
}

static double secuencial (void)
{
    double t0 = reloj();
    for (int i = 0; i < N; i++) r[i] = trabajo(i);
    return reloj() - t0;
}

/* Mejor de REP con la planificacion elegida en tiempo de ejecucion. */
static double paralelo(omp_sched_t tipo, int bloque)
{
    omp_set_schedule(tipo, bloque);
    double mejor = 1e30;
    for (int rep = 0; rep < REP; rep++) {
        double t0 = reloj();
        #pragma omp parallel for schedule (runtime)
        for (int i = 0; i < N; i++) r[i] = trabajo(i);
        double t = reloj() - t0;
        if (t < mejor) mejor = t;
    }
    return mejor;
}

/* EJERCICIO 2: cuanto trabajo se lleva cada hilo con esa planificacion.
Ejecuta el bucle real con schedule(runtime), suma el coste de cada
iteracion (vale i) en la casilla del hilo que la ejecuta, imprime cuantas
veces la media se lleva el mas cargado y cuantos hilos no han trabajado,
y devuelve ese factor. */
static double reparto (omp_sched_t tipo, int bloque)
{
    omp_set_schedule(tipo, bloque);
    long long carga[MAXH] = {0};

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        r[i] = trabajo(i);
        carga[omp_get_thread_num()] += i;
    }

    int p = omp_get_max_threads();
    long long total_trabajo = 0;

    for (int k = 0; k < p; k++) {
        total_trabajo += carga[k];
    }

    double media = (double)total_trabajo / p;
    long long max_carga = 0;
    int hilos_sin_trabajo = 0;

    for (int k = 0; k < p; k++) {
        if (carga[k] > max_carga) {
            max_carga = carga[k];
        }
        if (carga[k] == 0) {
            hilos_sin_trabajo++;
        }
    }

    double factor = (media > 0) ? ((double)max_carga / media) : 0.0;

    printf("mas cargado %.2f veces la media   hilos sin trabajo %d\n", factor, hilos_sin_trabajo);
    
    // Imprimir los datos detallados por hilo usando %I64d para Windows
    printf("  --> Desglose hilos: ");
    for (int k = 0; k < p; k++) {
        printf("[Hilo %d: %I64d] ", k, carga[k]);
    }
    printf("\n");
    return factor;
}

int main(void)
{
    struct { const char *nombre; omp_sched_t tipo; int bloque; } p[] = {
        {"static (por defecto)", omp_sched_static, 0},
        {"static, 1",          omp_sched_static, 1},
        {"dynamic (bloque 1)", omp_sched_dynamic, 1},
        {"dynamic, 64",        omp_sched_dynamic, 64},
        {"guided",             omp_sched_guided,  0},
    };

    printf("%d hilos, %d iteraciones, mejor de %d\n", omp_get_max_threads(), N, REP);

    double ts = 1e30;
    for (int rep = 0; rep < REP; rep++) { 
        double t = secuencial(); 
        if (t < ts) ts = t; 
    }

    double control = 0.0;
    for (int i = 0; i < N; i++) control += r[i];

    printf("%-22s %8.4f s %5.2fx\n", "secuencial", ts, 1.0);

    for (int k = 0; k < 5; k++) {
        double t = paralelo(p[k].tipo, p[k].bloque);
        printf("%-22s %8.4f s %5.2fx\n", p[k].nombre, t, ts/t);
    }

    double comprueba = 0.0;
    for (int i = 0; i < N; i++) comprueba += r[i];

    printf("resultado %s\n", (comprueba == control) ? "identico al secuencial" : "DISTINTO");

    printf("reparto del trabajo entre los %d hilos\n", omp_get_max_threads());
    for (int k = 0; k < 5; k++) {
        printf("%-22s ", p[k].nombre);
        reparto(p[k].tipo, p[k].bloque);
    }

    return 0;
}