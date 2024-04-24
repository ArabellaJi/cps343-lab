/*
 * $Smake: gcc -Wall -O3 -o %F %f -lcblas -latlas -lhdf5
 *
 * Computes a matrix-matrix product
 */

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

/* Macro to index matrices in column-major (Fortran) order */
#define IDX(i,j,stride) ((i)+(j)*(stride))  /* column major */

/* Check return values from HDF5 routines */
#define CHKERR(status,name) if (status) \
     fprintf(stderr, "Warning: nonzero status (%d) in %s\n", status, name)

/*----------------------------------------------------------------------------
 * Display string showing how to run program from command line
 *
 * Input:
 *   char* program_name (in)  name of executable
 * Output:
 *   writes to stderr
 * Returns:
 *   nothing
 */
void usage(char* program_name)
{
    fprintf(stderr, "Usage: %s [-v] input-file output-file\n", program_name);
}

/*----------------------------------------------------------------------------
 * Dump Matrix
 *
 * Parameters:
 *   double* a          (in)  pointer to matrix data
 *   int rows           (in)  number of rows in matrix
 *   int cols           (in)  number of columns in matrix
 *   int stride         (in)  =rows if column major or =cols if row major
* Returns:
 *   nothing
 */
void dumpMatrix(double* a, int rows, int cols, int stride)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf(" %8.2f", a[IDX(i,j,stride)]);
        }
        printf("\n");
    }
    printf("\n");
}

/*----------------------------------------------------------------------------
 * Create Matrix based on supplied name
 *
 * Parameters:
 *   char* name         (in)  name of matrix ("A" or "B")
 *   double** a         (out) pointer to pointer to matrix data
 *   int* rows          (out) pointer to number of rows
 *   int* cols          (out) pointer to number of cols
 * Returns:
 *   nothing
 */
void createMatrix(char* name, double** a, int* rows, int* cols)
{
    if (strcmp(name, "A") == 0)
    {
        *rows = 4;
        *cols = 2;
        *a = (double*) malloc(*rows * *cols * sizeof(double));
        (*a)[IDX(0,0,*rows)] =  4.0;
        (*a)[IDX(1,0,*rows)] =  2.0;
        (*a)[IDX(2,0,*rows)] = -2.0;
        (*a)[IDX(3,0,*rows)] =  1.0;
        (*a)[IDX(0,1,*rows)] = -4.0;
        (*a)[IDX(1,1,*rows)] = -1.0;
        (*a)[IDX(2,1,*rows)] = -3.0;
        (*a)[IDX(3,1,*rows)] =  4.0;
    }
    else if (strcmp(name, "B") == 0)
    {
        *rows = 2;
        *cols = 3;
        *a = (double*) malloc(*rows * *cols * sizeof(double));
        (*a)[IDX(0,0,*rows)] =  5.0;
        (*a)[IDX(1,0,*rows)] = -3.0;
        (*a)[IDX(0,1,*rows)] = -4.0;
        (*a)[IDX(1,1,*rows)] =  1.0;
        (*a)[IDX(0,2,*rows)] =  2.0;
        (*a)[IDX(1,2,*rows)] = -3.0;
    }
}

/*----------------------------------------------------------------------------
 * Form matrix product C = AB
 *
 * Parameters:
 *   double* c          (out) pointer to result matrix (nrows_a x ncols_b)
 *   double* a          (in)  pointer to left matrix
 *   int nrow_a         (in)  rows in left matrix
 *   int ncol_a         (in)  cols in left matrix (rows in right matrix)
 *   double* b          (in)  pointer to right matrix
 *   int ncol_b         (in)  cols in right matrix
 * Returns:
 *   nothing
 */
void matmat_jki(double* c, double* a, int nrow_a, int ncol_a,
                 double* b, int ncol_b)
{
    const int nrow_b = ncol_a;
    const int nrow_c = nrow_a;
    for (int j = 0; j < ncol_b; j++)
    {
        for (int i = 0; i < nrow_a; i++) c[IDX(i,j,nrow_c)] = 0.0;
        for (int k = 0; k < ncol_a; k++)
            for (int i = 0; i < nrow_a; i++)
                c[IDX(i,j,nrow_c)] += a[IDX(i,k,nrow_a)] * b[IDX(k,j,nrow_b)];
    }
}

/*----------------------------------------------------------------------------
 * Write Matrix
 */
void writeMatrix(char* fname, char* name, double* a, int rows, int cols)
{
    hid_t   file_id, group_id, dataspace_id, dataset_id;
    hsize_t dims[2];
    herr_t  status;

    /* Create HDF5 file.  If file already exists, truncate it */
    file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create matrix group */
    group_id = H5Gcreate(file_id, "/Matrix", H5P_DEFAULT, H5P_DEFAULT, 
                         H5P_DEFAULT);

    /* Create the data space for dataset */
    dims[0] = cols; /* assumes column-major storage pattern */
    dims[1] = rows;
    dataspace_id = H5Screate_simple(2, dims, NULL);

    /* Create the dataset */
    dataset_id = H5Dcreate(file_id, name, H5T_IEEE_F64LE, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Write matrix data to file */
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, a); CHKERR(status, "H5Dwrite()");

    /* Close resources */
    status = H5Dclose(dataset_id); CHKERR(status, "H5Dclose()");
    status = H5Sclose(dataspace_id); CHKERR(status, "H5Sclose()");
    status = H5Gclose(group_id); CHKERR(status, "H5Gclose()");
    status = H5Fclose(file_id); CHKERR(status, "H5Fclose()");
}

/*----------------------------------------------------------------------------
 * Read Matrix
 */
void readMatrix(char* fname, char* name, double** a, int* rows, int* cols)
{
    hid_t   file_id, dataset_id, file_dataspace_id, dataspace_id;
    herr_t status;
    hsize_t* dims;
    int rank;
    int ndims;
    hsize_t num_elem;

    /* Open existing HDF5 file */
    file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open existing first dataset */
    dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);

    /* Determine dataset parameters */
    file_dataspace_id = H5Dget_space(dataset_id);
    rank = H5Sget_simple_extent_ndims(file_dataspace_id);
    dims = (hsize_t*) malloc(rank * sizeof(hsize_t));
    ndims = H5Sget_simple_extent_dims(file_dataspace_id, dims, NULL);
    if (ndims != rank)
    {
        fprintf(stderr, "Warning: expected dataspace to be dimension ");
        fprintf(stderr, "%d but appears to be %d\n", rank, ndims);
    }

    /* Allocate matrix */
    num_elem = H5Sget_simple_extent_npoints(file_dataspace_id);
    *a = (double*) malloc(num_elem * sizeof(double));
    *cols = dims[0]; /* reversed since we're using Fortran-style ordering */
    *rows = dims[1];

    /* Create dataspace */
    dataspace_id = H5Screate_simple(rank, dims, NULL);

    /* Read matrix data from file */
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id,
                     file_dataspace_id, H5P_DEFAULT, *a);
    CHKERR(status, "H5Dread()");

    /* Close resources */
    status = H5Sclose(dataspace_id); CHKERR(status, "H5Sclose()");
    status = H5Sclose(file_dataspace_id); CHKERR(status, "H5Sclose()");
    status = H5Dclose(dataset_id); CHKERR(status, "H5Dclose()");
    status = H5Fclose(file_id); CHKERR(status, "H5Fclose()");
    free(dims);
}

/*----------------------------------------------------------------------------
 * Main program
 */
int main(int argc, char* argv[])
{
    char* in_name;
    char* out_name;
    double* a;             /* left matrix */
    double* b;             /* right matrix */
    double* c;             /* product C = AB */
    int nrow_a, ncol_a;    /* dimensions of left matrix */
    int nrow_b, ncol_b;    /* dimensions of right matrix */
    int nrow_c, ncol_c;    /* dimensions of product matrix */
    int verbose = 0;       /* nonzero for extra output */

    /* Process command line */
    int ch;
    while ((ch = getopt(argc, argv, "v")) != -1)
    {
        switch (ch)
        {
            case 'v':
                verbose++;
                break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }
    argv[optind - 1] = argv[0];
    argv += (optind - 1);
    argc -= (optind - 1);

    /* Make sure there are no additional arguments */
    if (argc != 3)
    {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    in_name  = argv[1];
    out_name = argv[2];

    /* read matrix data and optionally display it */
    readMatrix(in_name, "/Matrix/A", &a, &nrow_a, &ncol_a);
    readMatrix(in_name, "/Matrix/B", &b, &nrow_b, &ncol_b);

    if (ncol_a != nrow_b)
    {
        fprintf(stderr, "Error: matrix dimensions are not compatible\n");
        return EXIT_FAILURE;
    }

    if (verbose)
    {
        printf("Matrix A:\n");
        dumpMatrix(a, nrow_a, ncol_a, nrow_a);
        printf("Matrix B:\n");
        dumpMatrix(b, nrow_b, ncol_b, nrow_b);
    }

    /* Compute matrix product C = AB and display it */
    nrow_c = nrow_a;
    ncol_c = ncol_b;
    c = (double*) malloc(nrow_c * ncol_c * sizeof(double));

    // matmat_jki(c, a, nrow_a, ncol_a, b, ncol_b);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrow_a, ncol_b, ncol_a, 1, a, nrow_a, b, nrow_b, 0, c, nrow_c);
    if (verbose > 0)
    {
        printf("Matrix C = AB:\n");
        dumpMatrix(c, nrow_c, ncol_c, nrow_c);
    }

    /* Write product matrix to file */
    writeMatrix(out_name, "/Matrix/C", c, nrow_c, ncol_b);

    /* Clean up and quit */
    free(a);
    free(b);
    free(c);
    return 0;
}