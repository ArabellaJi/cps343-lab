/*
 * Thrust version
 *
 * $Smake: g++ -O2 -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -I/usr/local/cuda/include -o pi_thrust_omp pi_thrust.cc wtime.c
 *
 * Usage:
 *    pi_thrust_omp [-n NUM_SAMPLES] [-m SAMPLES_PER_THREAD] [-q]
 *
 * where the [] indicate an optional parameter.  Default values for the 
 * optional parameters are provided.
 *
 * This program is released into the public domain.
 *
 * ==========================================================================
 *
 * Estimate Pi using Monte Carlo sampling.  This is done using a 1x1 square
 * and a quarter circle of radius 1.  
 *
 *                    1 ----------------------------
 *                      |* * * *                   |
 *                      |        * *               |
 *                      |            * *           |
 *                      |                 *        |
 *                      |                   *      |
 *                      |                     *    |
 *                      |                     *    |
 *                      |                       *  |
 *                      |                       *  |
 *                      |                         *|
 *                      |                         *|
 *                      |                         *|
 *                      |                         *|
 *                    0 ----------------------------
 *                      0                          1
 *
 * The area of the square is 1 and the area the quarter circle is Pi/4.  The
 * ratio of these values should be the same as the ratio of samples inside
 * the quarter circle to the total number of samples, so
 *
 *             samples inside quarter circle     Pi/4
 *             ----------------------------- =  ----  = Pi/4
 *                total number of samples         1
 *
 * Thus, Pi can be estimated as
 *
 *                     4 (samples inside quarter circle)
 *               Pi ~= ---------------------------------
 *                        (total numeber of samples)
 *
 * ==========================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <unistd.h>
#include "wtime.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/functional.h>

//----------------------------------------------------------------------------
// Display results with or without labels
//
// Input:   long numSamples  - numbers of samples
//          double estimate  - estimate of Pi
//          double wtime     - elapsed wall-clock time
//          bool noLabels    - don't show labels if true
//
// Returns: nothing

void displayResults(long numSamples, double estimate, double wtime,
                    bool noLabels)
{
    const char* format = 
        (noLabels ?
         "%12.10f %10.3e %10.6f %ld\n" :
         "Pi: %12.10f, error: %10.3e, seconds: %g, samples: %ld\n");
    printf(format, estimate, M_PI - estimate, wtime, numSamples);
}

struct genRandomSamples
{
    const unsigned int seed;
    const long samplesPerThread;
    genRandomSamples (unsigned int _seed, long _samplesPerThread) : seed(_seed) , samplesPerThread(_samplesPerThread) {}
    __host__ __device__ 
        long operator()(int segment) const {
            long count = 0L;
            // Set up random number generator to provide uniformly
            // distributed pseudorandom numbers in the range [0.0, 1.0)
            thrust::default_random_engine rng(seed);
            thrust::uniform_real_distribution<double> uniDist(0.0, 1.0);
            // Skip parts of the sequence of random numbers used by prior threads
            rng.discard(segment * samplesPerThread * 2);

            for (long i = 0L; i < samplesPerThread; i++)
            {
                const double x = uniDist(rng);
                const double y = uniDist(rng);
                count += (x * x + y * y < 1.0 ? 1 : 0);
            }

            return count;
        }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    long numSamples = 50000000L;
    long samplesPerThread = 100L;
    bool quiet = false;

    // Process command line
    int c;
    while ((c = getopt(argc, argv, "n:m:q")) != -1)
    {
        switch(c)
        {
            case 'n':
                numSamples = atol(optarg);
                if (numSamples <= 0)
                {
                    fprintf(stderr, "number of samples must be positive\n");
                    fprintf(stderr, "got: %ld\n", numSamples);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'm':
                samplesPerThread = atol(optarg);
                if (samplesPerThread <= 0)
                {
                    fprintf(stderr, "samples per thread must be positive\n");
                    fprintf(stderr, "got: %ld\n", samplesPerThread);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'q':
                quiet = true;
                break;
            default:
                fprintf(stderr, "usage: %s [-n NUM_SAMPLES] [-m SAMPLES_PER_THREAD] [-q]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    numSamples = ((numSamples - 1 + samplesPerThread) / samplesPerThread) * samplesPerThread;

    // Get samples and compute estimate of Pi
    double t1 = wtime();
    long count = thrust::transform_reduce(
        thrust::counting_iterator<long>(0),
        thrust::counting_iterator<long>(numSamples / samplesPerThread),
        genRandomSamples(time(NULL), samplesPerThread),
        0,
        thrust::plus<long>()
        );
    double result = 4.0 * count / numSamples;
    double t2 = wtime();

    // Display result
    displayResults(numSamples, result, t2 - t1, quiet);

    // All done
    return 0;
}
