/*
 * $Smake: gcc -o %F %f; gcc -DARGS_REQUIRED -o %F_req %f
 *
 * Demonstration of passing integer and floating point parameters on
 * command line.
 *
 * Compile with
 *     gcc -o args args.c
 * or
 *     gcc -DARGS_REQUIRED -o args args.c
 *
 * If compiled with -DARGS_REQUIRED the two arguments must appear
 * on the command line when the program is run.  If compiled without it,
 * the arguments are optional and either neither, n, or both n and x may
 * appear on the command line.
 */

#include <stdio.h>
#include <stdlib.h>

const int    N_MIN =  100000;
const int    N_MAX = 1000000;

int main(int argc, char **argv)
{
    int    n = 100000;     /* default value */

#if defined(ARGS_REQUIRED)
    /*
     * we require two arguments on command line (argv[0] is program name)
     */
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s N\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    n = atol(argv[1]);
#else
    /*
     * both arguments are optional but if x is specified n must also appear
     */
    if (argc > 1) n = atol(argv[1]);
#endif

    /*
     * make sure input is valid
     */
    if (n < N_MIN || N_MAX <= n)
    {
        fprintf(stderr, "n = %d but must satisfy %d <= n < %d\n", 
                n, N_MIN, N_MAX);
        exit(EXIT_FAILURE);
    }

    /*
     * okay, we're good to go!
     */
    printf("n = %d\n", n);

    return 0;
}