#include<stdio.h>
#include<mpi.h>
#include<math.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int my_rank, num_procs;
    long long int i, n = 0;
    double sum, width, local, mypi, pi;
    double start = 0.0, stop = 0.0;
    int proc_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Get_processor_name(processor_name, &proc_len);
    printf("1111111111Process00 %d of %d\n", my_rank, num_procs);
    n = 20000000000;
    width = 1.0 / n;
    sum = 0.0;
    for (i = my_rank; i < n; i += num_procs) {
        //for (int j = 0; j < 10; j++)local += 1;
        local = width * ((double) i + 0.5);
        sum += 4.0 / (1.0 + local * local);
    }
    mypi = width * sum;
    printf("Process %d done\n", my_rank);

    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (my_rank == 0) {
        printf("PI is %.20f\n", pi);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
