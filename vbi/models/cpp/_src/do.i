
%module do

%{
#include "do.hpp"
%}

%include stl.i
%include "std_string.i"
/* instantiate the required template specializations */
namespace std {
    %template(IntVector)     vector<int>;
    %template(DoubleVector)  vector<double>;
    %template(DoubleVector2) vector<vector<double> >;
}

%include "do.hpp"
