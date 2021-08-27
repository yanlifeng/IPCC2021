#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include "omp.h"
#include <time.h>

int threadNumber = 64;

int main(int argc, char *argv[]) {


    long long int i, n = 0;
    double sum, width, mypi, pi;
    double start = 0.0, stop = 0.0;

    int my_rank, num_procs;
    int proc_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Get_processor_name(processor_name, &proc_len);
    printf("Process %d of %d ,processor name is %s\n", my_rank, num_procs,processor_name);


    int st = clock();
    n = 20000000000;
    width = 1.0 / n;
    sum = 0.0;
#pragma omp parallel for num_threads(threadNumber) reduction(+:sum)
    for (i = my_rank; i < n; i += num_procs) {
        //for (int j = 0; j < 10; j++)local += 1;
        double local = width * ((double) i + 0.5);
        sum += 4.0 / (1.0 + local * local);
    }
    mypi = width * sum;
    printf("Process %d done\n", my_rank);

    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (my_rank % 20 == 0) {
        printf("cost %.4f\n", (clock() - st) / 1e6);
    }
    if (my_rank == 0) {
        printf("PI is %.20f\n", pi);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
