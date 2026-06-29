/**
 * @file utility.hpp
 * @brief Common type aliases, utility functions, and RNG helpers shared by all
 *        VBI C++ model backends.
 *
 * Provides:
 *   - 1-D / 2-D vector type aliases (dim1, dim2, …)
 *   - Adjacency-matrix to adjacency-list conversion
 *   - Mathematical helpers (moving average, matrix–vector multiply)
 *   - Cross-platform memory and wall-clock timing
 *   - A seeded/random Mersenne-Twister RNG (rng())
 *   - File / folder existence checks and matrix I/O
 */

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <vector>
#include <string>
#include <random>
#include <assert.h>
#include <iostream>
#include <algorithm>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
    #include <io.h>
    #include <direct.h>
    #include <sys/stat.h>
    #pragma comment(lib, "psapi.lib")
    // Windows doesn't define S_IFDIR in the same way
    #ifndef S_IFDIR
        #define S_IFDIR _S_IFDIR
    #endif
#else
    #include <sys/stat.h>
    #include <unistd.h>
    #include <sys/time.h>
    #include <sys/resource.h>
#endif

// #include <Eigen/Dense>

using std::string;
using std::vector;

/** @brief 1-D vector of doubles. */
typedef std::vector<double> dim1;
/** @brief 1-D vector of unsigned integers. */
typedef std::vector<unsigned> dim1I;
/** @brief 2-D vector of doubles (vector of dim1). */
typedef std::vector<std::vector<double>> dim2;
/** @brief 2-D vector of unsigned integers. */
typedef std::vector<std::vector<unsigned>> dim2I;

/**
 * @brief Check whether a file exists on disk.
 * @param filename Path to the file.
 * @return true if the file exists, false otherwise.
 */
bool fileExists(const std::string &filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

/**
 * @brief Check whether a directory exists on disk.
 * @param path Path to the directory.
 * @return true if the path exists and is a directory, false otherwise.
 */
bool folderExists(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
    {
        if ((st.st_mode & S_IFDIR) != 0)
            return true;
        else
            return false;
    }
    else
        return false;
}

/**
 * @brief Convert a weighted adjacency matrix to a sparse adjacency list.
 *
 * Only edges with |A[i][j]| > 1e-8 are retained, which avoids iterating over
 * zero-weight entries in the inner simulation loops.
 *
 * @param A  Square N×N adjacency / weight matrix.
 * @return   adjlist[i] lists the indices j of all neighbours of node i.
 */
std::vector<std::vector<unsigned>> adjmat_to_adjlist(const dim2 &A)
{
    size_t n = A.size();

    std::vector<std::vector<unsigned>> adjlist;
    adjlist.resize(n);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            if (std::abs(A[i][j]) > 1e-8)
                adjlist[i].push_back(static_cast<unsigned>(j));
        }
    }

    return adjlist;
}

// std::vector<std::vector<size_t>> adjmat_to_adjlist(const dim2 &A)
// {
//     size_t n = A.size();

//     std::vector<std::vector<size_t>> adjlist;
//     adjlist.resize(n);

//     for (int i = 0; i < n; ++i)
//     {
//         for (int j = 0; j < n; ++j)
//         {
//             if (std::abs(A[j][i]) > 1e-8)
//                 adjlist[i].push_back(j);
//         }
//     }

//     return adjlist;
// }

/**
 * @brief Compute a non-overlapping block moving average.
 *
 * Partitions @p vec into consecutive blocks of size @p window and returns the
 * mean of each block.  The output length is floor(size / window).
 *
 * @param vec    Input time series.
 * @param window Block size (number of samples to average).
 * @return       Downsampled signal of length floor(vec.size() / window).
 */
dim1 moving_average(const dim1 &vec, const size_t window)
{
    size_t size = vec.size();
    size_t ind = 0;
    size_t buffer_size = size / window;
    dim1 vec_out(buffer_size);

    for (size_t itr = 0; itr < (size - window); ++itr)
    {
        double sum = 0.0;
        for (size_t j = itr; j < itr + window; ++j)
            sum += vec[j];
        vec_out[ind] = sum / double(window);
        ind++;
    }

    return vec_out;
}

/**
 * @brief Element-wise addition of two double vectors, writing the result into @p c.
 * @param a  First operand.
 * @param b  Second operand (same size as @p a).
 * @param c  Output vector (must be pre-allocated to the same size).
 */
void add(const vector<double> &a, const vector<double> &b, vector<double> &c)
{
    // c need to be allocated memory.
    transform(a.begin(), a.end(), b.begin(), c.begin(),
              [](double a, double b)
              { return a + b; });
}

/**
 * @brief Mixed-precision element-wise addition (float + double → float).
 * @param a  Float operand.
 * @param b  Double operand (same size as @p a).
 * @param c  Float output vector (pre-allocated).
 */
void add(const vector<float> &a, const vector<double> &b, vector<float> &c)
{
    assert(a.size() == b.size());
    assert(b.size() == c.size());

    for (size_t i = 0; i < a.size(); ++i)
        c[i] = a[i] + b[i];
}

/**
 * @brief Compute the arithmetic mean of a vector.
 * @param numbers  Input values.
 * @return         Mean, or 0 if the vector is empty.
 */
double average(std::vector<double> const &numbers)
{
    if (numbers.empty())
    {
        return 0;
    }
    double sum = 0;
    size_t arrayLength = numbers.size();
    for (size_t i = 0; i < arrayLength; i++)
        sum += numbers[i];
    return sum / arrayLength;

    // return std::reduce(v.begin(), v.end()) / count;
}

/**
 * @brief Compute the column-wise mean of a 2-D matrix V (shape nt × n).
 * @param V  Input matrix stored as vector-of-rows.
 * @return   dim1 of length n containing the mean over all rows.
 */
dim1 average(dim2 &V)
{
    // size_t num_nodes = (axis == "COL") ? V.size() : V[0].size();
    size_t n = V[0].size();
    size_t nt = V.size();
    dim1 out(n);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < nt; ++j)
            out[i] += V[j][i];
    }
    return out;
}

/**
 * @brief Return the peak resident-set size of the current process in kilobytes.
 *
 * Uses Windows PSAPI on Windows and getrusage on POSIX systems.
 *
 * @return Peak RSS in KB.
 */
long get_mem_usage()
{
    // measure memory usage
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
    {
        return pmc.WorkingSetSize / 1024; // Convert to KB to match Linux ru_maxrss
    }
    return 0;
#else
    struct rusage myusage;
    getrusage(RUSAGE_SELF, &myusage);
    return myusage.ru_maxrss;
#endif
}

/**
 * @brief Print elapsed wall time in h/min/s format.
 * @param wtime   Wall time in seconds.
 * @param cptime  CPU time in seconds (currently unused, reserved for future use).
 */
void display_timing(double wtime, double cptime)
{
    (void)cptime; // Mark as intentionally unused
    int wh;      //, ch;
    int wmin;    //, cpmin;
    double wsec; //, csec;
    wh = (int)wtime / 3600;
    // ch = (int)cptime / 3600;
    wmin = ((int)wtime % 3600) / 60;
    // cpmin = ((int)cptime % 3600) / 60;
    wsec = wtime - (3600. * wh + 60. * wmin);
    // csec = cptime - (3600. * ch + 60. * cpmin);
    printf("Wall Time : %d hours and %d minutes and %.4f seconds.\n", wh, wmin, wsec);
    // printf ("CPU  Time : %d hours and %d minutes and %.4f seconds.\n",ch,cpmin,csec);
}

/**
 * @brief Load a matrix from a whitespace-delimited text file.
 *
 * @tparam T       Element type (e.g., double, int).
 * @param filename Path to the text file.
 * @param row      Number of rows to read.
 * @param col      Number of columns to read.
 * @return         row × col matrix as a vector of vectors.
 *
 * Exits with code 2 if the file is not found.
 */
template <typename T>
inline std::vector<std::vector<T>> load_matrix(
    const std::string filename,
    const size_t row,
    const size_t col)
{
    /*!
    * Read matrix into vector of vector
    *
    * \param filename [string] name of text file to read
    * \param row [int] number of rows
    * \param col [int] number of columns

    * \return vector of vector of specified type
    *
    * **example**
    * std::vector<std::vector<int>> A = Neuro::read_matrix<int>(
    * "data/matrix_integer.txt", 4, 3);
    */

    std::ifstream ifile(filename);

    /*to check if input file exists*/
    if (fileExists(filename))
    {
        std::vector<std::vector<T>> Cij(row, std::vector<T>(col));

        for (size_t i = 0; i < row; i++)
        {
            for (size_t j = 0; j < col; j++)
            {
                ifile >> Cij[i][j];
            }
        }
        ifile.close();
        return Cij;
    }
    else
    {
        std::cerr << "\n file : " << filename << " not found \n";
        exit(2);
    }
}

/**
 * @brief Return the current wall-clock time in seconds.
 *
 * Uses QueryPerformanceCounter on Windows and gettimeofday on POSIX.
 *
 * @return Wall time in seconds (double precision).
 */
double get_wall_time()
{
    /*!
    measure real passed time
    \return wall time in second
    */
#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (QueryPerformanceFrequency(&frequency) && QueryPerformanceCounter(&counter))
    {
        return (double)counter.QuadPart / (double)frequency.QuadPart;
    }
    return 0.0;
#else
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
}

/**
 * @brief Return a reference to a shared Mersenne-Twister RNG instance.
 *
 * When @p fix_seed is true the generator is seeded with the constant 2,
 * ensuring reproducible noise realisations across calls.  When false, a
 * fresh seed from std::random_device is used instead.
 *
 * @param fix_seed  true → reproducible seed; false → random seed.
 * @return          Reference to a static std::mt19937 instance.
 */
std::mt19937 &rng(const bool fix_seed)
{
    if (fix_seed)
    {
        static std::mt19937 instance{2};
        return instance;
    }
    else
    {
        static std::mt19937 instance{std::random_device{}()};
        return instance;
    }
}

/**
 * @brief Set selected elements of a vector to a given value.
 * @param v        Vector to modify in-place.
 * @param indices  Indices of elements to set.
 * @param value    Value to assign at each index.
 */
void fill_vector(dim1 &v, const vector<int> indices, const double value)
{
    for (size_t i = 0; i < indices.size(); ++i)
        v[indices[i]] = value;
}

/**
 * @brief Check whether the last element of a vector is NaN.
 * @param vec  Input vector.
 * @return     0 if no NaN detected, -1 if the last element is NaN.
 */
int find_nan(const dim1 &vec)
{
    int ind = vec.size() - 1;
    if (std::isnan(vec[ind]))
    {
        std::cout << "nan found!" << std::endl;
        return -1;
    }
    return 0;
}

/**
 * @brief Dense matrix–vector product y = mat * x[offset:].
 *
 * @param mat     N×N weight matrix.
 * @param x       Input vector of length ≥ N + offset.
 * @param offset  Starting index in @p x (default 0).
 * @return        Result vector of length N.
 */
dim1 matvec(const dim2 &mat, const dim1 &x, const int offset = 0)
{
    int N = mat.size();
    dim1 y(N);

    // # pragma omp simd
    for (int i = 0; i < N; ++i)
    {
        y[i] = 0.0;
        for (int j = 0; j < N; ++j)
            y[i] += mat[i][j] * x[j + offset];
    }
    return y;
}

/**
 * @brief Sparse matrix–vector product using a pre-computed adjacency list.
 *
 * Only iterates over non-zero entries listed in @p a, improving performance
 * for sparse connectivity matrices.
 *
 * @param mat     N×N weight matrix (full storage, but only accessed at
 *                positions in @p a).
 * @param a       Adjacency list: a[i] contains the column indices j where
 *                mat[i][j] != 0.
 * @param x       Input vector of length ≥ N + offset.
 * @param offset  Starting index in @p x (default 0).
 * @return        Result vector of length N.
 */
dim1 matvec_s(const dim2 &mat, const dim2I &a, const dim1 &x, const int offset = 0)
{
    int n = mat.size();
    dim1 y(n);

    # pragma omp simd
    for (int i=0; i<n; ++i)
    {
        for(int j:a[i])
        {
            y[i] += mat[i][j] * x[j + offset];
        }
    }
    return y;
}

// dim1 matmul_e(const Eigen::MatrixXd &A, const Eigen::VectorXd &x)
// {
//     Eigen::VectorXd y = A * x;
//     dim1 y_vec(y.data(), y.data() + y.size());
//     return y_vec;
// }

#endif
