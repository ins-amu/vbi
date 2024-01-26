#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <vector>
#include <string>
#include <random>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
// #include <Eigen/Dense>

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<unsigned> dim1I;
typedef std::vector<std::vector<double>> dim2;
typedef std::vector<std::vector<unsigned>> dim2I;

bool fileExists(const std::string &filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

bool folderExists(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
    {
        if (st.st_mode & S_IFDIR != 0)
            return true;
        else
            return false;
    }
    else
        return false;
}

std::vector<std::vector<unsigned>> adjmat_to_adjlist(const dim2 &A)
{
    size_t n = A.size();

    std::vector<std::vector<unsigned>> adjlist;
    adjlist.resize(n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (std::abs(A[i][j]) > 1e-8)
                adjlist[i].push_back(j);
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

dim1 moving_average(const dim1 &vec, const size_t window)
{
    size_t size = vec.size();
    size_t ind = 0;
    size_t buffer_size = int(size / window);
    dim1 vec_out(buffer_size);

    for (size_t itr = 0; itr < (size - window); ++itr)
    {
        double sum = 0.0;
        for (int j = itr; j < itr + window; ++j)
            sum += vec[j];
        vec_out[ind] = sum / double(window);
        ind++;
    }

    return vec_out;
}

// adding vectors
void add(const vector<double> &a, const vector<double> &b, vector<double> &c)
{
    // c need to be allocated memory.
    transform(a.begin(), a.end(), b.begin(), c.begin(),
              [](double a, double b)
              { return a + b; });
}

void add(const vector<float> &a, const vector<double> &b, vector<float> &c)
{
    assert(a.size() == b.size());
    assert(b.size() == c.size());

    for (size_t i = 0; i < a.size(); ++i)
        c[i] = a[i] + b[i];
}

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

long get_mem_usage()
{
    // measure memory usage

    struct rusage myusage;

    getrusage(RUSAGE_SELF, &myusage);
    return myusage.ru_maxrss;
}

void display_timing(double wtime, double cptime)
{
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

double get_wall_time()
{
    /*!
    measure real passed time
    \return wall time in second
    */

    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

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

void fill_vector(dim1 &v, const vector<int> indices, const double value)
{
    for (size_t i = 0; i < indices.size(); ++i)
        v[indices[i]] = value;
}

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
