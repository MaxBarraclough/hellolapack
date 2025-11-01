
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
    std::cout << output_matrix[0] << ' ' << output_matrix[1] << ' ' << output_matrix[2] << '\n';
    std::cout << output_matrix[3] << ' ' << output_matrix[4] << ' ' << output_matrix[5] << '\n';
    std::cout << output_matrix[6] << ' ' << output_matrix[7] << ' ' << output_matrix[8] << '\n';
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

    std::cout << "Output of matrix multiplication using single-precision floats (block matrix example)\n";

    std::size_t idx = 0;
//  std::cout << std::left; // No good, causes left alignment and trailing whitespace
//  std::cout << std::internal; // Equivalent to right, for our purposes
//  std::cout << std::right; // This is the default
    for (std::size_t row_counter = 0; row_counter != square_matrix_order; ++row_counter)
    {
        for (std::size_t col_counter = 0; col_counter != square_matrix_order; ++col_counter)
        {
            std::cout << std::setw(8); // Must call this every time
            std::cout << output_matrix[idx];
            ++idx;
        }
        std::cout << '\n';
    }
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


    std::cout << "Output of matrix multiplication using floats:\n";
    std::cout << output_matrix[0] << ' ' << output_matrix[1] << ' ' << output_matrix[2] << '\n';
    std::cout << output_matrix[3] << ' ' << output_matrix[4] << ' ' << output_matrix[5] << '\n';
    std::cout << output_matrix[6] << ' ' << output_matrix[7] << ' ' << output_matrix[8] << '\n';
    std::cout << std::endl;
}



// Based on https://stackoverflow.com/a/3525136/
// Inverts a matrix in-place.
// Returns 0 if successful, non-zero otherwise. Never throws.
// Failure does not guarantee the matrix is unmodified.
[[nodiscard]] static lapack_int invert_matrix(double* A, lapack_int N) noexcept
{
    lapack_int status;

    try
    {
        lapack_int working_buffer_num_elements = N * N; // Assume no overflow

        std::unique_ptr<lapack_int[]> pivot_indices_uniqueptr
          = std::make_unique<lapack_int[]>(N);
        lapack_int *pivot_indices_raw = pivot_indices_uniqueptr.get();

    //  Array of double, with capacity MAX(1,working_buffer_num_elements)
        std::unique_ptr<double[]> working_buffer_uniqueptr
          = std::make_unique<double[]>(working_buffer_num_elements);
        double *working_buffer_raw = working_buffer_uniqueptr.get();

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


        // If input matrix is singular, status will now not equal zero
        if (status == 0) {
            // Second step: matrix inversion proper.
            // Documentation of the FORTRAN interface:
            // https://www.netlib.org/lapack/explore-html/da/d28/group__getri_ga8b6904853957cfb37eba1e72a3f372d2.html#ga8b6904853957cfb37eba1e72a3f372d2
            LAPACK_dgetri(
                &N,                           // lapack_int const* n
                A,                            // double*           A
                &N,                           // lapack_int const* lda
                pivot_indices_raw,            // lapack_int const* ipiv
                working_buffer_raw,           // double*           work
                &working_buffer_num_elements, // lapack_int const* lwork
                &status                       // lapack_int*       info
            );
        }
    }
    catch (...) // Must have been an allocation failure
    {
        status = std::numeric_limits< lapack_int >::min();
    }
    return status;
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
    std::cout << the_matrix[0] << ' ' << the_matrix[1]  << '\n';
    std::cout << the_matrix[2] << ' ' << the_matrix[3]  << '\n';
    std::cout << std::endl;

    if ( 0 == invert_matrix(the_matrix, square_matrix_order) )
    {
        std::cout << "Inverted matrix:\n";
        std::cout << the_matrix[0] << ' ' << the_matrix[1]  << '\n';
        std::cout << the_matrix[2] << ' ' << the_matrix[3]  << '\n';
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
        std::cout << mult_matrix[0] << ' ' << mult_matrix[1]  << '\n';
        std::cout << mult_matrix[2] << ' ' << mult_matrix[3]  << '\n';
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

    std::cout        << matrix2[0];
    std::cout << ' ' << matrix2[1];
    std::cout << ' ' << matrix2[2];
    std::cout << ' ' << matrix2[3];
    std::cout << ' ' << matrix2[4];
    std::cout << ' ' << matrix2[5];
    std::cout << ' ' << matrix2[6];
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
    std::cout << output_matrix[0]  << ' ' << output_matrix[1]  << ' ' << output_matrix[2]  << ' ' << output_matrix[3]  << '\n';
    std::cout << output_matrix[4]  << ' ' << output_matrix[5]  << ' ' << output_matrix[6]  << ' ' << output_matrix[7]  << '\n';
    std::cout << output_matrix[8]  << ' ' << output_matrix[9]  << ' ' << output_matrix[10] << ' ' << output_matrix[11] << '\n';
    std::cout << output_matrix[12] << ' ' << output_matrix[13] << ' ' << output_matrix[14] << ' ' << output_matrix[15] << '\n';
    std::cout << std::endl;
}



int main(int argc, char *argv[])
{
    // Assuming we're using OpenBLAS header file and binary:
//  std::cout << openblas_get_config() << '\n';
//  std::cout << std::endl;

//  use_sgemm();
//  use_dgemm();
//  use_invert_matrix();
//  use_dtbmv();
//  use_sgemm_on_binary_matrix();
    use_sgemm_block_matrices();

    return 0;
}

