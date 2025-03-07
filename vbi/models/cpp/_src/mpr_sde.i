
%module mpr_sde
%{
#include "mpr_sde.hpp"
%}

%include stl.i
%include <std_vector.i>
%include "std_string.i"
/* instantiate the required template specializations */
namespace std {
    %template(IntVector)     std::vector<int>;
    %template(DoubleVector)  std::vector<double>;
    %template(DoubleVector2) std::vector<vector<double> >;
    %template(SingleVector)  std::vector<float>;
    %template(SingleVector2) std::vector<vector<float> >;
}

%include "mpr_sde.hpp"
