
%module wc_ode

%{
#include "wc_ode.hpp"
%}

%include stl.i
%include "std_string.i"
/* instantiate the required template specializations */
namespace std {
    %template(IntVector)     vector<int>;
    %template(DoubleVector)  vector<double>;
    %template(DoubleVector2) vector<vector<double> >;
    %template(SingleVector)  vector<float>;
    %template(SingleVector2) vector<vector<float> >;
}

%include "wc_ode.hpp"
