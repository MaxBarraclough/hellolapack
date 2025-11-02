
//  ABOUT
//  =====
//
//  Small example program showing matrix multiplication using BLAS
//  (tested against OpenBLAS) and matrix inversion using OpenBLAS LAPACK


#include <cstddef> // for std::size_t
#include <iostream>
#include <iomanip> // For std::setw
#include <memory> // For std::unique_ptr
#include <array>  // For std::size which gives count of a raw array
#include <limits> // For std::numeric_limits

#include <lapacke.h>
#include <cblas.h> // Whichever cblas.h is the system default
// #include "cblas-openblas.h" // Specifically the OpenBLAS header

#define FMT_WIDTH 16


inline static void print_matrix_f(const std::size_t num_rows, const std::size_t num_cols, const float * const matrix)
{
    std::size_t idx = 0;
    for (std::size_t row_counter = 0; row_counter != num_rows; ++row_counter)
    {
        for (std::size_t col_counter = 0; col_counter != num_cols; ++col_counter)
        {
            std::cout << std::setw(FMT_WIDTH); // Must call this every time
            std::cout << matrix[idx];
            ++idx;
        }
        std::cout << '\n';
    }
    // We do not print any trailing newline
}

inline static void print_matrix_d(const std::size_t num_rows, const std::size_t num_cols, const double * const matrix)
{
    std::size_t idx = 0;
    for (std::size_t row_counter = 0; row_counter != num_rows; ++row_counter)
    {
        for (std::size_t col_counter = 0; col_counter != num_cols; ++col_counter)
        {
            std::cout << std::setw(FMT_WIDTH); // Must call this every time
            std::cout << matrix[idx];
            ++idx;
        }
        std::cout << '\n';
    }
    // We do not print any trailing newline
}



static void use_sgemm()
{
    constexpr std::size_t square_matrix_order = 3;

    constexpr std::size_t square_matrix_count
      = square_matrix_order * square_matrix_order;


    // Very small matrices, no need to use heap
    // Unpadded row-major representation for simplicity
    float matrix1[] =
    { 5.0f, 0.0f, 0.0f,
      0.0f, 5.0f, 0.0f,
      0.0f, 0.0f, 5.0f
    };
    // Recall std::size gives count, not size like sizeof
    static_assert( std::size(matrix1) == square_matrix_count );

    float matrix2[] =
    { 2.0f, 0.0f, 0.0f,
      0.0f, 2.0f, 0.0f,
      0.0f, 0.0f, 2.0f
    };
    static_assert( std::size(matrix2) == square_matrix_count );

    float output_matrix[square_matrix_count]; // = {0.0f};
    // https://www.netlib.org/clapack/cblas/dgemm.c
    // https://www.math.utah.edu/software/lapack/lapack-blas/dsymm.html
    // say that:
    //   When BETA is supplied as zero then C need not be set on input.

    // Recall that lda and ldb are for customising stride (e.g. padding)

    // Recall CBLAS_ORDER and CBLAS_LAYOUT are synonyms
    cblas_sgemm(
        CBLAS_ORDER::CblasRowMajor,    // CBLAS_LAYOUT layout
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransB

        square_matrix_order,           // const CBLAS_INDEX M
        square_matrix_order,           // const CBLAS_INDEX N
        square_matrix_order,           // const CBLAS_INDEX K

        1.0f,                          // const float alpha
        matrix1,                       // const float *A
        square_matrix_order,           // const CBLAS_INDEX lda

        matrix2,                       // const float *B
        square_matrix_order,           // const CBLAS_INDEX ldb
        0.0f,                          // const float beta

        output_matrix,                 // float *C
        square_matrix_order            // const CBLAS_INDEX ldc
    );

    std::cout << "Output of matrix multiplication using single-precision floats:\n";
    print_matrix_f(
        square_matrix_order, // num_rows
        square_matrix_order, // num_cols
        output_matrix        // matrix
    );
    std::cout << std::endl;
}



static void use_sgemm_block_matrices()
{
    constexpr std::size_t square_matrix_order = 6;

    constexpr std::size_t square_matrix_count
      = square_matrix_order * square_matrix_order;


    // Very small matrices, no need to use heap
    // Unpadded row-major representation for simplicity
    float matrix1[] =
    { 1.0f, 2.0f, 3.0f,   0.0f, 0.0f, 0.0f,
      4.0f, 5.0f, 6.0f,   0.0f, 0.0f, 0.0f,
      7.0f, 8.0f, 9.0f,   0.0f, 0.0f, 0.0f,

      0.0f, 0.0f, 0.0f,  10.0f,20.0f,30.0f,
      0.0f, 0.0f, 0.0f,  40.0f,50.0f,60.0f,
      0.0f, 0.0f, 0.0f,  70.0f,80.0f,90.0f
    };
    // Recall std::size gives count, not size like sizeof
    static_assert( std::size(matrix1) == square_matrix_count );

    float matrix2[] =
    {10.0f,40.0f,70.0f,   0.0f, 0.0f, 0.0f,
     20.0f,50.0f,80.0f,   0.0f, 0.0f, 0.0f,
     30.0f,60.0f,90.0f,   0.0f, 0.0f, 0.0f,

      0.0f, 0.0f, 0.0f,  1.0f, 4.0f, 7.0f,
      0.0f, 0.0f, 0.0f,  2.0f, 5.0f, 8.0f,
      0.0f, 0.0f, 0.0f,  3.0f, 6.0f, 9.0f
    };
    static_assert( std::size(matrix2) == square_matrix_count );

    float output_matrix[square_matrix_count]; // = {0.0f};
    // https://www.netlib.org/clapack/cblas/dgemm.c
    // https://www.math.utah.edu/software/lapack/lapack-blas/dsymm.html
    // say that:
    //   When BETA is supplied as zero then C need not be set on input.

    // Recall that lda and ldb are for customising stride (e.g. padding)

    // Recall CBLAS_ORDER and CBLAS_LAYOUT are synonyms
    cblas_sgemm(
        CBLAS_ORDER::CblasRowMajor,    // CBLAS_LAYOUT layout
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransB

        square_matrix_order,           // const CBLAS_INDEX M
        square_matrix_order,           // const CBLAS_INDEX N
        square_matrix_order,           // const CBLAS_INDEX K

        1.0f,                          // const float alpha
        matrix1,                       // const float *A
        square_matrix_order,           // const CBLAS_INDEX lda

        matrix2,                       // const float *B
        square_matrix_order,           // const CBLAS_INDEX ldb
        0.0f,                          // const float beta

        output_matrix,                 // float *C
        square_matrix_order            // const CBLAS_INDEX ldc
    );

    std::cout << "Output of multiplication of block matrices using single-precision floats:\n";

    print_matrix_f(
        square_matrix_order, // num_rows
        square_matrix_order, // num_cols
        output_matrix        // matrix
    );
    std::cout << std::endl;
}



static void use_dgemm()
{
    constexpr std::size_t use_dgemm_square_matrix_order = 3;

    constexpr std::size_t use_dgemm_square_matrix_count
      = use_dgemm_square_matrix_order * use_dgemm_square_matrix_order;


    double matrix1[] =
    { 5.0, 0.0, 0.0,
      0.0, 5.0, 0.0,
      0.0, 0.0, 5.0
    };
    static_assert( std::size(matrix1) == use_dgemm_square_matrix_count );

    double matrix2[] =
    { 2.0, 0.0, 0.0,
      0.0, 2.0, 0.0,
      0.0, 0.0, 2.0
    };
    static_assert( std::size(matrix2) == use_dgemm_square_matrix_count );

    double output_matrix[9]; // = {0.0};
    static_assert( std::size(output_matrix) == use_dgemm_square_matrix_count );


    cblas_dgemm(
        CBLAS_LAYOUT::CblasRowMajor,    // CBLAS_LAYOUT layout
        CBLAS_TRANSPOSE::CblasNoTrans,  // CBLAS_TRANSPOSE TransA
        CBLAS_TRANSPOSE::CblasNoTrans,  // CBLAS_TRANSPOSE TransB

        use_dgemm_square_matrix_order,  // const CBLAS_INDEX M
        use_dgemm_square_matrix_order,  // const CBLAS_INDEX N
        use_dgemm_square_matrix_order,  // const CBLAS_INDEX K

        1.0,                            // const double alpha
        matrix1,                        // const double *A
        use_dgemm_square_matrix_order,  // const CBLAS_INDEX lda

        matrix2,                        // const double *B
        use_dgemm_square_matrix_order,  // const CBLAS_INDEX ldb
        0.0,                            // const double beta

        output_matrix,                  // double *C
        use_dgemm_square_matrix_order   // const CBLAS_INDEX ldc
    );


    std::cout << "Output of matrix multiplication using doubles:\n";
    print_matrix_d(
        use_dgemm_square_matrix_order, // num_rows
        use_dgemm_square_matrix_order, // num_cols
        output_matrix                  // matrix
    );
    std::cout << std::endl;
}



// TODO optional buffer parameter.
// Based on https://stackoverflow.com/a/3525136/
// Inverts a square matrix, of extent N, in-place.
// Returns 0 if successful, non-zero otherwise (e.g. if matrix is singular).
// Never throws.
// Failure does not guarantee the matrix is unmodified.
[[nodiscard]] static lapack_int invert_matrix(double* A, lapack_int N) noexcept
{
    lapack_int status = -1;

    try
    {
        const lapack_int working_buffer_num_elements = N * N; // Assume no overflow

        const std::unique_ptr<lapack_int[]> pivot_indices_uniqueptr
          = std::make_unique<lapack_int[]>(N);
        lapack_int * const pivot_indices_raw = pivot_indices_uniqueptr.get();

    //  Array of double, with capacity MAX(1,working_buffer_num_elements)
        const std::unique_ptr<double[]> working_buffer_uniqueptr
          = std::make_unique<double[]>(working_buffer_num_elements);
        double * const working_buffer_raw = working_buffer_uniqueptr.get();

        // We can see the conventional style in:
        // https://www.netlib.org/lapack/lapacke.html

        // First stage: LU factorisation.
        // Documentation of the FORTRAN interface:
        // https://www.netlib.org/lapack/explore-html/db/d04/group__getrf_gaea332d65e208d833716b405ea2a1ab69.html#gaea332d65e208d833716b405ea2a1ab69
        LAPACK_dgetrf(
            &N,                // lapack_int const* m
            &N,                // lapack_int const* n
            A,                 // double*           A
            &N,                // lapack_int const* lda
            pivot_indices_raw, // lapack_int*       ipiv
            &status            // lapack_int*       info
        );
        // Per documentation, pivot_indices_raw is now populated. A now holds
        // the sum of the upper and lower triangular matrices, except that the
        // unit diagonal elements of the lower triangular matrix are discarded
        // prior to that sum.
        // If input matrix is singular, status will now be positive (nonzero).

        if (status == 0) {
            // Second step: matrix inversion proper.
            // Documentation of the FORTRAN interface:
            // https://www.netlib.org/lapack/explore-html/da/d28/group__getri_ga8b6904853957cfb37eba1e72a3f372d2.html#ga8b6904853957cfb37eba1e72a3f372d2
            LAPACK_dgetri(
                &N,                           // lapack_int const* n      (in)
                A,                            // double*           A      (in,out)
                &N,                           // lapack_int const* lda    (in)
                pivot_indices_raw,            // lapack_int const* ipiv   (in)
                working_buffer_raw,           // double*           work   (out)
                &working_buffer_num_elements, // lapack_int const* lwork  (in)
                &status                       // lapack_int*       info   (out)
            );
        }
    }
    catch (...) // Must have been an allocation failure
    {
        status = std::numeric_limits< lapack_int >::min();
    }
    return status;
}



static void polynomial_regresssion_using_lapack()
{

    // Recall it's a row per object, and a column per variable.
    // Recall also there must be strictly more rows (objects) than cols (vars).
    constexpr std::size_t design_matrix_num_rows    = 5;
    constexpr std::size_t design_matrix_num_columns = 3;

    constexpr std::size_t design_matrix_count
      = design_matrix_num_columns * design_matrix_num_rows;


// All 1s
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    1.0,
//    1.0,
//    1.0,
//    1.0
//  };

// Straight line of gradient 1 passing through the origin:
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    2.0,
//    3.0,
//    4.0,
//    5.0
//  };

// Like 'y=x^2'
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    4.0,
//    9.0,
//   16.0,
//   25.0
//  };

// Like 'y = (x+1)^2 + 10', i.e. 'y = x^2 + 2x + 1 + 10'
    double y_or_beta_vec[] = // Column vector
    { 4.0 + 10.0,
      9.0 + 10.0,
     16.0 + 10.0,
     25.0 + 10.0,
     36.0 + 10.0
    };

    static_assert( std::size(y_or_beta_vec) == design_matrix_num_rows );

//  // Makes more sense to hold the transpose
//  double the_design_matrix_transposed[] =
//  {
//      1.0,         1.0,         1.0,         1.0,         1.0,        // Always 1
//      1.0,         2.0,         3.0,         4.0,         5.0,        // X^1
//      (1.0 * 1.0), (2.0 * 2.0), (3.0 * 3.0), (4.0 * 4.0), (5.0 * 5.0) // X^2
//  };
//  static_assert( std::size(the_design_matrix_transposed) == design_matrix_count );

    double the_design_matrix[] =
    { // Always 1 | x^1  | x^2
        1.0,        1.0,   (1.0 * 1.0),
        1.0,        2.0,   (2.0 * 2.0),
        1.0,        3.0,   (3.0 * 3.0),
        1.0,        4.0,   (4.0 * 4.0),
        1.0,        5.0,   (5.0 * 5.0)
    };
    static_assert( std::size(the_design_matrix) == design_matrix_count );


// https://www.netlib.org/lapack/explore-html/d8/d83/group__gels_gaa65298f8ef218a625e40d0da3c95803c.html#gaa65298f8ef218a625e40d0da3c95803c
// After execution, B shall hold the following (from the docs):
//   if TRANS = 'N' and m >= n, rows 1 to n of B contain the least squares
//   solution vectors; the residual sum of squares for the solution in each
//   column is given by the sum of squares of elements N+1 to M in that column

    const lapack_int result =
      LAPACKE_dgels(
          LAPACK_ROW_MAJOR,             // int matrix_layout // Also works: CBLAS_LAYOUT::CblasRowMajor
          'N',                          // char trans   Do not try: CBLAS_TRANSPOSE::CblasNoTrans
          design_matrix_num_rows,       // lapack_int m     // Number of rows in matrix A
          design_matrix_num_columns,    // lapack_int n     // Number of columns in matrix A
          1,                            // lapack_int nrhs  // Number of columns in matrices B and X
          the_design_matrix,            // double* a        // Full of garbage afterward (for our purposes)
          design_matrix_num_columns,    // lapack_int lda
          y_or_beta_vec,                // double* b        // Input holding B, output as described above
          1                             // lapack_int ldb
      );

    if (0 == result)
    {
        std::cout << "Final beta vector:\n";
        print_matrix_d(
            design_matrix_num_columns, // num_rows
            1,                         // num_cols
            y_or_beta_vec              // matrix
        );
        std::cout << std::endl;
    }
    else
    {
        std::cerr << "Error encountered. There is no output to print.";
        std::cerr << std::endl;
    }

}


// Unoptimised variant
static void use_invert_matrix_for_polynomial_regresssion_unoptimized()
{
    // Recall it's a row per object, and a column per variable.
    // Recall also there must be strictly more rows (objects) than cols (vars).
    constexpr std::size_t design_matrix_num_rows    = 5;
    constexpr std::size_t design_matrix_num_columns = 3;

    constexpr std::size_t design_matrix_count
      = design_matrix_num_columns * design_matrix_num_rows;


// All 1s
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    1.0,
//    1.0,
//    1.0,
//    1.0
//  };

// Straight line of gradient 1 passing through the origin:
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    2.0,
//    3.0,
//    4.0,
//    5.0
//  };

// Like 'y=x^2'
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    4.0,
//    9.0,
//   16.0,
//   25.0
//  };

// Like 'y = (x+1)^2 + 10', i.e. 'y = x^2 + 2x + 1 + 10'
    double the_y_vector[] = // Column vector
    { 4.0 + 10.0,
      9.0 + 10.0,
     16.0 + 10.0,
     25.0 + 10.0,
     36.0 + 10.0
    };

    static_assert( std::size(the_y_vector) == design_matrix_num_rows );

    // Or, equivalently, the design matrix in column-major representation
    double the_design_matrix_transposed[] =
    {
        1.0,         1.0,         1.0,         1.0,         1.0,        // Always 1
        1.0,         2.0,         3.0,         4.0,         5.0,        // X^1
        (1.0 * 1.0), (2.0 * 2.0), (3.0 * 3.0), (4.0 * 4.0), (5.0 * 5.0) // X^2
    };
    static_assert( std::size(the_design_matrix_transposed) == design_matrix_count );

    double the_design_matrix[] =
    { // Always 1 | x^1  | x^2
        1.0,        1.0,   (1.0 * 1.0),
        1.0,        2.0,   (2.0 * 2.0),
        1.0,        3.0,   (3.0 * 3.0),
        1.0,        4.0,   (4.0 * 4.0),
        1.0,        5.0,   (5.0 * 5.0)
    };
    static_assert( std::size(the_design_matrix) == design_matrix_count );

    // Compute product of transpose of design matrix, and design matrix.
    // Later this array will hold the inverse of that matrix.
    double left_matrix_or_inverse[design_matrix_count] = {0.0};

    // TODO optimise away the explicit transposed matrix
    cblas_dgemm(
        CBLAS_LAYOUT::CblasRowMajor,   // CBLAS_LAYOUT layout
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransB

        design_matrix_num_columns,     // const CBLAS_INDEX M // Num rows, matrix A
        design_matrix_num_columns,     // const CBLAS_INDEX N // Num cols, matrix B
        design_matrix_num_rows,        // const CBLAS_INDEX K // Num cols of matrix A and num rows of B

        1.0,                           // const double alpha
        the_design_matrix_transposed,  // const double *A
        design_matrix_num_rows,        // const CBLAS_INDEX lda
         // i.e. num cols in the transposed matrix

        the_design_matrix,             // const double *B
        design_matrix_num_columns,     // const CBLAS_INDEX ldb
        0.0,                           // const double beta

        left_matrix_or_inverse,        // double *C
        design_matrix_num_columns      // const CBLAS_INDEX ldc
    );


    // Print left_matrix_or_inverse (not yet inverted).
    // Recall it's a symmetric square matrix.
    std::cout << "Printing left_matrix_or_inverse (not yet inverted):\n";
    print_matrix_d(
        design_matrix_num_columns, // num_rows
        design_matrix_num_columns, // num_cols
        left_matrix_or_inverse     // matrix
    );
    std::cout << std::endl;

    if ( 0 == invert_matrix(left_matrix_or_inverse, design_matrix_num_columns) )
    {
        std::cout << "Printing left_matrix_or_inverse (now inverted):\n";
        print_matrix_d(
            design_matrix_num_columns, // num_rows
            design_matrix_num_columns, // num_cols
            left_matrix_or_inverse     // matrix
        );
        std::cout << std::endl;

        // Now compute product of transposed design matrix, and the y column vector.
        // This gives a column vector with the same number of rows as the transposed design matrix,
        // i.e. the number of columns in the design matrix.
        {
            double rhs_col_vec[design_matrix_num_columns] = {0.0};

            cblas_dgemm(
                CBLAS_LAYOUT::CblasRowMajor,   // CBLAS_LAYOUT layout
                CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
                CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransB

                design_matrix_num_columns,     // const CBLAS_INDEX M // Num rows, matrix A
                1,                             // const CBLAS_INDEX N // Num cols, matrix B
                design_matrix_num_rows,        // const CBLAS_INDEX K // Num cols of matrix A and num rows of B

                1.0,                           // const double alpha
                the_design_matrix_transposed,  // const double *A
                design_matrix_num_rows,        // const CBLAS_INDEX lda
                 // i.e. num cols in the transposed matrix

                the_y_vector,                  // const double *B
                1,                             // const CBLAS_INDEX ldb
                0.0,                           // const double beta

                rhs_col_vec,                   // double *C
                1                              // const CBLAS_INDEX ldc
            );

            std::cout << "Product of transposed design matrix, and the y column vector:\n";
            print_matrix_d(
                design_matrix_num_columns, // num_rows
                1,                         // num_cols
                rhs_col_vec                // matrix
            );
            std::cout << std::endl;

            // Final product, of the inverted matrix and the right-hand-side
            // column vector. This gives a column vector.
            // Num of rows equals that of the inverted matrix
            // (recall it's a square matrix of extent design_matrix_num_columns).
            // TODO can we reuse buffers? Reduce numbers of buffers we allocate?
            {
                double beta_col_vec[design_matrix_num_columns] = {0.0};

                cblas_dgemm(
                    CBLAS_LAYOUT::CblasRowMajor,   // CBLAS_LAYOUT layout
                    CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
                    CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransB

                    design_matrix_num_columns,     // const CBLAS_INDEX M // Num rows, matrix A
                    1,                             // const CBLAS_INDEX N // Num cols, matrix B
                    design_matrix_num_columns,     // const CBLAS_INDEX K // Num cols of matrix A and num rows of B

                    1.0,                           // const double alpha
                    left_matrix_or_inverse,        // const double *A
                    design_matrix_num_columns,     // const CBLAS_INDEX lda

                    rhs_col_vec,                   // const double *B
                    1,                             // const CBLAS_INDEX ldb
                    0.0,                           // const double beta

                    beta_col_vec,                  // double *C
                    1                              // const CBLAS_INDEX ldc
                );

                std::cout << "Final beta vector:\n";
                print_matrix_d(
                    design_matrix_num_columns, // num_rows
                    1,                         // num_cols
                    beta_col_vec               // matrix
                );
                std::cout << std::endl;
            }
        }
    }
    else
    {
        std::cerr << "Matrix inversion failed. Was the matrix singular?";
        std::cerr << std::endl;
    }
}



static void use_invert_matrix_for_polynomial_regresssion_optimized()
{
    // Recall it's a row per object, and a column per variable.
    // Recall also there must be strictly more rows (objects) than cols (vars).
    constexpr std::size_t design_matrix_num_rows    = 5;
    constexpr std::size_t design_matrix_num_columns = 3;

    constexpr std::size_t design_matrix_count
      = design_matrix_num_columns * design_matrix_num_rows;


// All 1s
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    1.0,
//    1.0,
//    1.0,
//    1.0
//  };

// Straight line of gradient 1 passing through the origin:
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    2.0,
//    3.0,
//    4.0,
//    5.0
//  };

// Like 'y=x^2'
//  double the_y_vector[] = // Column vector
//  { 1.0,
//    4.0,
//    9.0,
//   16.0,
//   25.0
//  };

// Like 'y = (x+1)^2 + 10', i.e. 'y = x^2 + 2x + 1 + 10'
    double the_y_vector[] = // Column vector
    { 4.0 + 10.0,
      9.0 + 10.0,
     16.0 + 10.0,
     25.0 + 10.0,
     36.0 + 10.0
    };

    static_assert( std::size(the_y_vector) == design_matrix_num_rows );

    // Makes more sense to hold the transpose
    double the_design_matrix_transposed[] =
    {
        1.0,         1.0,         1.0,         1.0,         1.0,        // Always 1
        1.0,         2.0,         3.0,         4.0,         5.0,        // X^1
        (1.0 * 1.0), (2.0 * 2.0), (3.0 * 3.0), (4.0 * 4.0), (5.0 * 5.0) // X^2
    };
    static_assert( std::size(the_design_matrix_transposed) == design_matrix_count );


    // Compute product of transpose of design matrix, and design matrix.
    // Later this array will hold the inverse of that matrix.
    double left_matrix_or_inverse[design_matrix_count] = {0.0};

    // TODO optimise away the explicit transposed matrix
    cblas_dgemm(
        CBLAS_LAYOUT::CblasRowMajor,   // CBLAS_LAYOUT layout
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
        CBLAS_TRANSPOSE::CblasTrans,   // CBLAS_TRANSPOSE TransB // Take the transpose of the transposed matrix

        design_matrix_num_columns,     // const CBLAS_INDEX M // Num rows, matrix A
        design_matrix_num_columns,     // const CBLAS_INDEX N // Num cols, matrix B
        design_matrix_num_rows,        // const CBLAS_INDEX K // Num cols of matrix A and num rows of B

        1.0,                           // const double alpha
        the_design_matrix_transposed,  // const double *A
        design_matrix_num_rows,        // const CBLAS_INDEX lda
         // i.e. num cols in the transposed matrix

        the_design_matrix_transposed,  // const double *B
        design_matrix_num_rows,        // const CBLAS_INDEX ldb  // Transpose, so ...rows not ...columns
        0.0,                           // const double beta

        left_matrix_or_inverse,        // double *C
        design_matrix_num_columns      // const CBLAS_INDEX ldc
    );


    // Print left_matrix_or_inverse (not yet inverted).
    // Recall it's a symmetric square matrix.
    std::cout << "Printing left_matrix_or_inverse (not yet inverted):\n";
    print_matrix_d(
        design_matrix_num_columns, // num_rows
        design_matrix_num_columns, // num_cols
        left_matrix_or_inverse     // matrix
    );
    std::cout << std::endl;


    // Here we don't take advantage of the matrix being symmetric:
    if ( 0 == invert_matrix(left_matrix_or_inverse, design_matrix_num_columns) )
    {
        std::cout << "Printing left_matrix_or_inverse (now inverted):\n";
        print_matrix_d(
            design_matrix_num_columns, // num_rows
            design_matrix_num_columns, // num_cols
            left_matrix_or_inverse     // matrix
        );
        std::cout << std::endl;

        // Now compute product of transposed design matrix, and the y column vector.
        // This gives a column vector with the same number of rows as the transposed design matrix,
        // i.e. the number of columns in the design matrix.
        {
            double rhs_col_vec[design_matrix_num_columns] = {0.0};

            cblas_dgemm(
                CBLAS_LAYOUT::CblasRowMajor,   // CBLAS_LAYOUT layout
                CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
                CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransB

                design_matrix_num_columns,     // const CBLAS_INDEX M // Num rows, matrix A
                1,                             // const CBLAS_INDEX N // Num cols, matrix B
                design_matrix_num_rows,        // const CBLAS_INDEX K // Num cols of matrix A and num rows of B

                1.0,                           // const double alpha
                the_design_matrix_transposed,  // const double *A
                design_matrix_num_rows,        // const CBLAS_INDEX lda
                 // i.e. num cols in the transposed matrix

                the_y_vector,                  // const double *B
                1,                             // const CBLAS_INDEX ldb
                0.0,                           // const double beta

                rhs_col_vec,                   // double *C
                1                              // const CBLAS_INDEX ldc
            );

            std::cout << "Product of transposed design matrix, and the y column vector:\n";
            print_matrix_d(
                design_matrix_num_columns, // num_rows
                1,                         // num_cols
                rhs_col_vec                // matrix
            );
            std::cout << std::endl;

            // Final product, of the inverted matrix and the right-hand-side
            // column vector. This gives a column vector.
            // Num of rows equals that of the inverted matrix
            // (recall it's a square matrix of extent design_matrix_num_columns).
            // TODO can we reuse buffers? Reduce numbers of buffers we allocate?
            {
                double beta_col_vec[design_matrix_num_columns] = {0.0};

                // Matrix left_matrix_or_inverse is symmetric, so use the specialised function:
                cblas_dsymm( // returns void
                    CBLAS_LAYOUT::CblasRowMajor, // OPENBLAS_CONST enum CBLAS_ORDER Order,
                    CBLAS_SIDE::CblasLeft,       // OPENBLAS_CONST enum CBLAS_SIDE Side,     Left matrix (A) is symmetric, not B
                    CBLAS_UPLO::CblasUpper,      // OPENBLAS_CONST enum CBLAS_UPLO Uplo,
                    design_matrix_num_columns,   // OPENBLAS_CONST blasint M,                Num rows in target matrix C
                    1,                           // OPENBLAS_CONST blasint N,                Num columns in target matrix C
                    1.0,                         // OPENBLAS_CONST double alpha,
                    left_matrix_or_inverse,      // OPENBLAS_CONST double *A,
                    design_matrix_num_columns,   // OPENBLAS_CONST blasint lda,
                    rhs_col_vec,                 // OPENBLAS_CONST double *B,
                    1,                           // OPENBLAS_CONST blasint ldb,
                    0.0,                         // OPENBLAS_CONST double beta,
                    beta_col_vec,                // double *C,
                    1                            // OPENBLAS_CONST blasint ldc
                );

                std::cout << "Final beta vector:\n";
                print_matrix_d(
                    design_matrix_num_columns, // num_rows
                    1,                         // num_cols
                    beta_col_vec               // matrix
                );
            }
        }
    }
    else
    {
        std::cerr << "Matrix inversion failed. Was the matrix singular?";
        std::cerr << std::endl;
    }
}



static void use_invert_matrix()
{
    constexpr std::size_t square_matrix_order = 2;

    constexpr std::size_t square_matrix_count
      = square_matrix_order * square_matrix_order;


    double the_matrix[] =
    { 4.0, 7.0,
      2.0, 6.0
    };
//  { 1.0, 1.0, // A singular matrix
//    0.0, 0.0
//  };
    static_assert( std::size(the_matrix) == square_matrix_count );

    double the_original_matrix[] =
    { 4.0, 7.0,
      2.0, 6.0
    };
//  { 1.0, 1.0, // A singular matrix
//    0.0, 0.0
//  };


    std::cout << "Matrix to invert:\n";
    print_matrix_d(
        square_matrix_order, // num_rows
        square_matrix_order, // num_cols
        the_matrix           // matrix
    );
    std::cout << std::endl;

    if ( 0 == invert_matrix(the_matrix, square_matrix_order) )
    {
        std::cout << "Inverted matrix:\n";
        print_matrix_d(
            square_matrix_order, // num_rows
            square_matrix_order, // num_cols
            the_matrix           // matrix
        );
        std::cout << std::endl;

        // Multiply the matrices, hopefully arriving at an identity matrix
        // Inputs: the_matrix, the_original_matrix
        // Output: mult_matrix

        double mult_matrix[square_matrix_count];

        cblas_dgemm(
            CBLAS_LAYOUT::CblasRowMajor,   // CBLAS_LAYOUT layout
            CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
            CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransB

            square_matrix_order,           // const CBLAS_INDEX M
            square_matrix_order,           // const CBLAS_INDEX N
            square_matrix_order,           // const CBLAS_INDEX K

            1.0,                           // const double alpha
            the_matrix,                    // const double *A
            square_matrix_order,           // const CBLAS_INDEX lda

            the_original_matrix,           // const double *B
            square_matrix_order,           // const CBLAS_INDEX ldb
            0.0,                           // const double beta

            mult_matrix,                   // double *C
            square_matrix_order            // const CBLAS_INDEX ldc
        );

        std::cout << "This should be an identity matrix:\n";
        print_matrix_d(
            square_matrix_order, // num_rows
            square_matrix_order, // num_cols
            mult_matrix          // matrix
        );
        std::cout << std::endl;
    }
    else
    {
        std::cerr << "Error inverting matrix" << std::endl;
    }
}



static void use_dtbmv()
{
/*
https://www.netlib.org/lapack/explore-html/d6/d9f/group__tbmv_ga0d7fd7b684cb64944eee8b29e8809272.html#ga0d7fd7b684cb64944eee8b29e8809272

    DTBMV  performs one of the matrix-vector operations

       x := A*x,   or   x := A**T*x,

    where x is an n element vector and A is an n by n unit, or non-unit,
    upper or lower triangular band matrix, with ( k + 1 ) diagonals.

Not to be confused with dgbmv which works with band matrices that need not be
upper or lower diagonal.

We can use dtbmv to do elementwise multiplication between 2 vectors, i.e.
Hadamard product.

This kind of trick probably isn't a good idea in real world code.
The function isn't intended for this use and is likely to perform much slower
than the naive approach, see related:
https://stackoverflow.com/questions/7621520/element-wise-vector-vector-multiplication-in-blas#comment112278219_13433038
*/

    const std::size_t matrix_order = 7;

    double matrix1[] = {
          1.1,
          2.2,
          3.3,
          4.4,
          5.5,
          6.6,
          7.7 };
    static_assert( std::size(matrix1) == matrix_order );

    double matrix2[] = {
          1.0,
         10.0,
        100.0,
       1000.0,
      10000.0,
     100000.0,
    1000000.0 };
    static_assert( std::size(matrix2) == matrix_order );

    cblas_dtbmv(                       // void cblas_dtbmv(
        CBLAS_LAYOUT::CblasRowMajor,   // CBLAS_LAYOUT layout
        CBLAS_UPLO::CblasUpper,        // CBLAS_UPLO Uplo
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransA
        CBLAS_DIAG::CblasNonUnit,      // CBLAS_DIAG Diag
        matrix_order,                  // const CBLAS_INDEX N // Order of matrix A
        0,                             // const CBLAS_INDEX K // Number of superdiagonals/subdiagonals
        matrix1,                       // const double *A
        1,                             // const CBLAS_INDEX lda // First dimension of A, i.e. stride
        matrix2,                       // double *X
        1                              // const CBLAS_INDEX incX // The increment for elements of X, i.e. stride
    );                                 // );

    std::cout << "Output of multiplying diagonal matrix by column matrix:\n";
    print_matrix_d(
        matrix_order, // num_rows
        1,            // num_cols
        matrix2       // matrix
    );
    std::cout << std::endl;
}



static void use_sgemm_on_binary_matrix()
{
    constexpr std::size_t matrix_num_cols =  4;
    constexpr std::size_t matrix_num_rows = 10;

    constexpr std::size_t matrix_element_count
        = matrix_num_cols * matrix_num_rows;

    // Very small matrices, no need to use heap
    // Unpadded row-major representation for simplicity
    const float matrix1[] =
    { /* 1  */   1,      1,      0,      0,
      /* 2  */   1,      0,      1,      0,
      /* 3  */   1,      0,      0,      1,
      /* 4  */   1,      0,      0,      0,
      /* 5  */   0,      1,      0,      1,
      /* 6  */   0,      1,      1,      0,
      /* 7  */   0,      1,      0,      0,
      /* 8  */   0,      0,      1,      1,
      /* 9  */   0,      0,      1,      0,
      /* 10 */   0,      0,      0,      1,
    };

    // Recall std::size gives count, not size like sizeof
    static_assert( std::size(matrix1) == matrix_element_count );

//  float matrix2[] =
//  { 2.0f, 0.0f, 0.0f,
//    0.0f, 2.0f, 0.0f,
//    0.0f, 0.0f, 2.0f
//  };
//  static_assert( std::size(matrix2) == square_matrix_count );

    float output_matrix[matrix_element_count] = {0.0f};
    // https://www.netlib.org/clapack/cblas/dgemm.c
    // https://www.math.utah.edu/software/lapack/lapack-blas/dsymm.html
    // say that:
    //   When BETA is supplied as zero then C need not be set on input.

    // Recall that lda and ldb are for customising stride (e.g. padding)

// From https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga8cad871c590600454d22564eff4fed6b.html#ga8cad871c590600454d22564eff4fed6b :
//
//     SGEMM  performs one of the matrix-matrix operations
//
//        C := alpha*op( A )*op( B ) + beta*C,
//
//     where  op( X ) is one of
//
//        op( X ) = X   or   op( X ) = X**T,
//
//     alpha and beta are scalars, and A, B and C are matrices, with:
//       op( A ) an m by k matrix
//       op( B ) a  k by n matrix
//       C an m by n matrix

// TODO try this using LAPACKE_sgemqr
    // Recall CBLAS_ORDER and CBLAS_LAYOUT are synonyms.
    // Recall also that cblas_sgemm returns NULL.

// if (false)
    cblas_sgemm(
        CBLAS_ORDER::CblasRowMajor,    // CBLAS_LAYOUT layout
        CBLAS_TRANSPOSE::CblasTrans,   // CBLAS_TRANSPOSE TransA
        CBLAS_TRANSPOSE::CblasNoTrans, // CBLAS_TRANSPOSE TransB

        matrix_num_cols,               // const CBLAS_INDEX M
        matrix_num_cols,               // const CBLAS_INDEX N
        matrix_num_rows,               // const CBLAS_INDEX K

        1.0f,                          // const float alpha

        matrix1,                       // const float *A
        matrix_num_cols,               // const CBLAS_INDEX lda

        matrix1,                       // const float *B
        matrix_num_cols,               // const CBLAS_INDEX ldb

        0.0f,                          // const float beta

        output_matrix,                 // float *C
        matrix_num_cols                // const CBLAS_INDEX ldc
    );


/*
Signature:

lapack_int LAPACKE_sgemqrt(
    int          matrix_layout,
    char         side,
    char         trans,
    lapack_int   m,
    lapack_int   n,
    lapack_int   k,
    lapack_int   nb,
    const float* v,
    lapack_int   ldv,
    const float* t,
    lapack_int   ldt,
    float*       c,
    lapack_int   ldc
);

*/
#if 0
lapack_int result_status = LAPACKE_sgemqrt(
                         // int          matrix_layout,
                         // char         side,
                         // char         trans,
                         // lapack_int   m,
                         // lapack_int   n,
                         // lapack_int   k,
                         // lapack_int   nb,
                         // const float* v,
                         // lapack_int   ldv,
                         // const float* t,
                         // lapack_int   ldt,
                         // float*       c,
                         // lapack_int   ldc
);
#endif

// LAPACKE_sgemm; // Does not exist

    std::cout << "Output of matrix multiplication using doubles:\n";
    print_matrix_f(
        matrix_num_rows, // num_rows
        matrix_num_cols, // num_cols
        output_matrix    // matrix
    );
    std::cout << std::endl;
}



// int main(int argc, char *argv[])
int main() // Avoid compiler warnings as argc and argv are unused
{
    // Assuming we're using OpenBLAS header file and binary:
//  std::cout << openblas_get_config() << '\n';
//  std::cout << std::endl;

    std::cout << "---------------------------------\n"
                 "-- Basic matrix multiplication --\n"
                 "--           (float)           --\n"
                 "---------------------------------\n";
    use_sgemm();


    std::cout << "---------------------------------\n"
                 "-- Basic matrix multiplication --\n"
                 "--          (double)           --\n"
                 "---------------------------------\n";
    use_dgemm();


    std::cout << "---------------------------------\n"
                 "--       Matrix inversion      --\n"
                 "---------------------------------\n";
    use_invert_matrix();


    std::cout << "--------------------------------------------------\n"
                 "-- Multiplying diagonal matrix by column vector --\n"
                 "--------------------------------------------------\n";
    use_dtbmv();


    std::cout << "----------------------------------------------\n"
                 "-- Matrix multiplication with binary matrix --\n"
                 "----------------------------------------------\n";
    use_sgemm_on_binary_matrix();


    std::cout << "-----------------------------------------------\n"
                 "-- Matrix multiplication with block matrices --\n"
                 "-----------------------------------------------\n";
    use_sgemm_block_matrices();

    std::cout << "----------------------------------------\n"
                 "-- Polynomial regression, unoptimised --\n"
                 "----------------------------------------\n";
    use_invert_matrix_for_polynomial_regresssion_unoptimized();

    std::cout << "-------------------------------------------\n"
                 "-- Polynomial regression, more optimised --\n"
                 "-------------------------------------------\n";

    use_invert_matrix_for_polynomial_regresssion_optimized();

    std::cout << "-------------------------------------------------\n"
                 "-- Polynomial regression with LAPACKE function --\n"
                 "-------------------------------------------------\n";
    polynomial_regresssion_using_lapack();

    return 0;
}

