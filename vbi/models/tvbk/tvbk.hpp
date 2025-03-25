#pragma once
#include <stdint.h>
#include <math.h>
#include <vector>
#include <thread>

/*
                                             gg            
                                             ""            
  ,ggg,,ggg,,ggg,     ,gggg,gg    ,gggg,gg   gg     ,gggg, 
 ,8" "8P" "8P" "8,   dP"  "Y8I   dP"  "Y8I   88    dP"  "Yb
 I8   8I   8I   8I  i8'    ,8I  i8'    ,8I   88   i8'      
,dP   8I   8I   Yb,,d8,   ,d8b,,d8,   ,d8I _,88,_,d8,_    _
8P'   8I   8I   `Y8P"Y8888P"`Y8P"Y8888P"8888P""Y8P""Y8888PP
                                      ,d8I'                
                                    ,dP'8I                 
                                   ,8"  8I                 
                                   I8   8I                 
                                   `8, ,8I                 
                                    `Y8P"  
*/
// at -O3 -fopen-simd, these kernels result in compact asm
#ifdef _MSC_VER
#include <intrin.h>
#define INLINE __forceinline
#define kernel(name, expr, ...) \
template <int width> INLINE static void name (__VA_ARGS__) \
{ \
  _Pragma("loop(hint_parallel(8))") \
  _Pragma("loop(ivdep)") \
  for (int i=0; i < width; i++) \
    expr;\
}
#else
#define INLINE __attribute__((always_inline)) inline
#define kernel(name, expr, args...) \
template <int width> INLINE static void name (args) \
{ \
  _Pragma("clang loop unroll(full)") \
  for (int i=0; i < width; i++) \
    expr;\
}
#endif

// MSVC and others may not define M_PI?
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define M_PI_F ((float) M_PI)

namespace tvbk {

  /*
         8I               I8              
         8I               I8              
         8I            88888888           
         8I               I8              
   ,gggg,8I    ,gggg,gg   I8     ,gggg,gg 
  dP"  "Y8I   dP"  "Y8I   I8    dP"  "Y8I 
 i8'    ,8I  i8'    ,8I  ,I8,  i8'    ,8I 
,d8,   ,d8b,,d8,   ,d8b,,d88b,,d8,   ,d8b,
P"Y8888P"`Y8P"Y8888P"`Y88P""Y8P"Y8888P"`Y8 
  */

template <int width>
struct cxb {
  /* values for 1st and 2nd Heun stage respectively.
     each shaped (num_node, ) */
  float *cx1;
  float *cx2;
  /* delay buffer (num_node, num_time)*/
  float *buf;
  const uint32_t num_node;
  const uint32_t num_time; // horizon, power of 2
  const uint32_t num_item=width;
  cxb(const uint32_t num_node, const uint32_t num_time)
      : num_node(num_node), num_time(num_time), cx1(new float[num_node * width]),
        cx2(new float[num_node * width]), buf(new float[num_node * num_time * width]) {}
  // init from existing buffers
  cxb(const uint32_t num_node, const uint32_t num_time, float *cx1, float *cx2, float *buf)
      : num_node(num_node), num_time(num_time), cx1(cx1), cx2(cx2), buf(buf) {}
};

template <int width> struct cxbs  {
  float *cx1, *cx2, *buf;
  const uint32_t num_node, num_time, num_item=width, num_batch;
  cxbs(const uint32_t num_node, const uint32_t num_time, const uint32_t num_batches)
      : num_node(num_node), num_time(num_time), num_batch(num_batches),
        cx1(new float[num_node * width * num_batches]), cx2(new float[num_node * width * num_batches]),
        buf(new float[num_node * num_time * width * num_batches]) {}
  const cxb<width> batch(const uint32_t i) const {
    return cxb<width>(this->num_node, this->num_time, this->cx1 + i * this->num_node * width,
                      this->cx2 + i * this->num_node * width, this->buf + i * this->num_node * this->num_time * width);
  }
};

typedef cxb<1> cx;
typedef cxb<8> cx8; // common case: 8-wide SIMD
typedef cxbs<8> cx8s;

struct conn {
  const uint32_t num_node;
  const uint32_t num_nonzero;
  const float *weights;    // (num_nonzero,)
  const uint32_t *indices; // (num_nonzero,)
  const uint32_t *indptr;  // (num_nodes+1,)
  const uint32_t *idelays; // (num_nonzero,)
  conn(const uint32_t num_node, const uint32_t num_nonzero)
      : num_node(num_node), num_nonzero(num_nonzero),
        weights(new float[num_nonzero]), indices(new uint32_t[num_nonzero]),
        indptr(new uint32_t[num_node + 1]), idelays(new uint32_t[num_nonzero]) {
  }
};

/*
                                              8I                                
                                              8I                                
                                              8I                                
                                              8I                                
  ,gggggg,    ,gggg,gg   ,ggg,,ggg,     ,gggg,8I    ,ggggg,    ,ggg,,ggg,,ggg,  
  dP""""8I   dP"  "Y8I  ,8" "8P" "8,   dP"  "Y8I   dP"  "Y8ggg,8" "8P" "8P" "8, 
 ,8'    8I  i8'    ,8I  I8   8I   8I  i8'    ,8I  i8'    ,8I  I8   8I   8I   8I 
,dP     Y8,,d8,   ,d8b,,dP   8I   Yb,,d8,   ,d8b,,d8,   ,d8' ,dP   8I   8I   Yb,
8P      `Y8P"Y8888P"`Y88P'   8I   `Y8P"Y8888P"`Y8P"Y8888P"   8P'   8I   8I   `Y8
*/

INLINE uint64_t sfc64(uint64_t s[4]) {
  uint64_t r = s[0] + s[1] + s[3]++;
  s[0] = (s[1] >> 11) ^ s[1];
  s[1] = (s[2] << 3) + s[2];
  s[2] = r + (s[2] << 24 | s[2] >> 40);
  return r;
}

INLINE float randn1(uint64_t s[4]) {
  uint64_t u = sfc64(s);
#ifdef _MSC_VER
  // TODO check, cf https://stackoverflow.com/a/42913358
  double x = __popcnt64(u >> 32);
#else
  double x = __builtin_popcount(u >> 32);
#endif
  x += (uint32_t)u * (1 / 4294967296.);
  x -= 16.5;
  x *= 0.3517262290563295;
  return (float)x;
}

INLINE void randn(uint64_t *seed, int n, float *out) {
  #pragma omp simd
  for (int i = 0; i < n; i++)
    out[i] = randn1(seed);
}

template <int width>
INLINE void krandn(uint64_t **seed, float *out) {
  #pragma omp simd
  for (int i = 0; i < width; i++)
    out[i] = randn1(seed[i]);
}
/*
 ,dPYb,                                                ,dPYb,          
 IP'`Yb                                                IP'`Yb          
 I8  8I                                                I8  8I          
 I8  8bgg,                                             I8  8'          
 I8 dP" "8   ,ggg,    ,gggggg,   ,ggg,,ggg,    ,ggg,   I8 dP    ,g,    
 I8d8bggP"  i8" "8i   dP""""8I  ,8" "8P" "8,  i8" "8i  I8dP    ,8'8,   
 I8P' "Yb,  I8, ,8I  ,8'    8I  I8   8I   8I  I8, ,8I  I8P    ,8'  Yb  
,d8    `Yb, `YbadP' ,dP     Y8,,dP   8I   Yb, `YbadP' ,d8b,_ ,8'_   8) 
88P      Y8888P"Y8888P      `Y88P'   8I   `Y8888P"Y8888P'"Y88P' "YY8P8P 
*/


/* simple stuff */
kernel(inc,      x[i]                    += w*y[i], float *x, float *y, float w)
kernel(adds,     x[i]                         += a, float *x, float a)
kernel(load,     x[i]                       = y[i], float *x, float *y)
kernel(zero,     x[i]                        = 0.f, float *x)
kernel(muls,     x[i]                       *=   w, float *x, float w)

/* Heun stages */
kernel(heunpred, xi[i]           = x[i] + dt*dx[i], float *x, float *xi, float *dx, float dt)
kernel(heuncorr, x[i] += dt*0.5f*(dx1[i] + dx2[i]), float *x, float *dx1, float *dx2, float dt)
kernel(sheunpred, xi[i]           = x[i] + dt*dx[i] + z[i], float *x, float *xi, float *dx, float *z, float dt)
kernel(sheuncorr, x[i] += dt*0.5f*(dx1[i] + dx2[i]) + z[i], float *x, float *dx1, float *dx2, float *z, float dt)

/* activation functions */
kernel(sigm,     x[i] = 1.0f/(1.0f + expf(y[i])), float *x, float *y)
kernel(heavi,    x[i] = y[i] >= 0.0f ? 1.0f : 0.0f, float *x, float *y)
kernel(relu,     x[i] = y[i] >= 0.0f ? y[i] : 0.0f, float *x, float *y)
kernel(lrelu,    x[i] = y[i] >= 0.0f ? y[i] : 0.01f*y[i], float *x, float *y)

/* transcendentals; vectorized by gcc w/ libmvec;
   need sleef or similar elsewhere */
kernel(kfabsf, x[i] = fabsf(y[i]), float *x, float *y)
kernel(klogf,  x[i] = logf(y[i]), float *x, float *y)
kernel(kpowfp,  x[i] = powf(y[i], z[i]), float *x, float *y, float *z)
kernel(kpowf,  x[i] = powf(y[i], z), float *x, float *y, float z)
kernel(kexpf,  x[i] = expf(y[i]), float *x, float *y)
kernel(kexp2f, x[i] = exp2f(y[i]), float *x, float *y)
kernel(ksqrtf, x[i] = sqrtf(y[i]), float *x, float *y)
kernel(ksinf,  x[i] = sinf(y[i]), float *x, float *y)
kernel(kcosf,  x[i] = cosf(y[i]), float *x, float *y)
kernel(ktanf,  x[i] = tanf(y[i]), float *x, float *y)
kernel(kerff,  x[i] = erff(y[i]), float *x, float *y)

/* short length dot product accumulates, so doesn't fit into
   macro above */
template <int width>
INLINE static void dot(float *dst, float *x, float *y)
{
    float acc=0.f;
    #pragma omp simd reduction(+:acc)
    for (int i=0; i<width; i++) acc+=x[i]*y[i];
    *dst = acc;
}

/*
                         I8                                        ,dPYb,    
                         I8                                        IP'`Yb    
                      88888888                                     I8  8I    
                         I8                                        I8  8bgg, 
  ,ggg,,ggg,    ,ggg,    I8   gg    gg    gg   ,ggggg,   ,gggggg,  I8 dP" "8 
 ,8" "8P" "8,  i8" "8i   I8   I8    I8    88bgdP"  "Y8gggdP""""8I  I8d8bggP" 
 I8   8I   8I  I8, ,8I  ,I8,  I8    I8    8I i8'    ,8I ,8'    8I  I8P' "Yb, 
,dP   8I   Yb, `YbadP' ,d88b,,d8,  ,d8,  ,8I,d8,   ,d8',dP     Y8,,d8    `Yb,
8P'   8I   `Y8888P"Y8888P""Y8P""Y88P""Y88P" P"Y8888P"  8P      `Y888P      Y8       
*/
INLINE void cx_all_j(const cx &cx, const conn &c, uint32_t t, uint32_t j) {
  uint32_t wrap_mask = cx.num_time - 1; // assume num_time is power of 2
  float *const buf = cx.buf + j * cx.num_time;
  uint32_t th = t + cx.num_time;
#pragma omp simd
  for (uint32_t l = c.indptr[j]; l < c.indptr[j + 1]; l++) {
    uint32_t i = c.indices[l];
    float w = c.weights[l];
    uint32_t d = c.idelays[l];
    uint32_t p1 = (th - d) & wrap_mask;
    uint32_t p2 = (th - d + 1) & wrap_mask;
    cx.cx1[i] += w * buf[p1];
    cx.cx2[i] += w * buf[p2];
  }
}

INLINE void cx_j(const cx &cx, const conn &c, uint32_t t) {
#pragma omp simd
  for (int i = 0; i < c.num_node; i++)
    cx.cx1[i] = cx.cx2[i] = 0.0f;
  for (int j = 0; j < c.num_node; j++)
    cx_all_j(cx, c, t, j);
}

INLINE void cx_i(const cx &cx, const conn &c, uint32_t t) {
  uint32_t wrap_mask = cx.num_time - 1; // assume num_time is power of 2
  uint32_t th = t + cx.num_time;
#pragma omp simd
  for (int i = 0; i < c.num_node; i++) {
    float cx1 = 0.f, cx2 = 0.f;
    for (uint32_t l = c.indptr[i]; l < c.indptr[i + 1]; l++) {
      uint32_t j = c.indices[l];
      float w = c.weights[l];
      uint32_t d = c.idelays[l];
      uint32_t p1 = (th - d) & wrap_mask;
      uint32_t p2 = (th - d + 1) & wrap_mask;
      cx1 += w * cx.buf[j * cx.num_time + p1];
      cx2 += w * cx.buf[j * cx.num_time + p2];
    }
    cx.cx1[i] = cx1;
    cx.cx2[i] = cx2;
  }
}

// setup pointers b1 & b2 to delay_buffer to read from
template <int width=8>
static void INLINE prep_ij(
  const cxb<width> &cx, const conn &c,
  const int i_time, const int nz, float **b1, float **b2, float *w)
{
  uint32_t H = cx.num_time, Hm1 = H - 1;
  *w = c.weights[nz];
  float *b0 = cx.buf + H * c.indices[nz] * width;
  int t0 = H + i_time - c.idelays[nz];
  // TODO keep copy of t=0 at t=H so we don't need to do two
  // modulos and two pointers: just needs an extra write every H
  // time steps.
  *b1 = b0 + ((t0 + 0) & Hm1) * width;
  *b2 = b0 + ((t0 + 1) & Hm1) * width;
}

  // vectorized prep_ij but poitners are 8 bytes, so only do 4 at time
  // TODO switch to width instead of 4 fixed
template <int width=8>
  static void INLINE prep4_ij(
  const cxb<width> &cx, const conn &c,
  const int i_time, const int nz, float **b1, float **b2, float *w)
{
  uint32_t H = cx.num_time, Hm1 = H - 1;
  const float *weights = c.weights + nz;
  const uint32_t *indices = c.indices + nz;
  const uint32_t *idelays = c.idelays + nz;
  float *buf = cx.buf;
#pragma clang loop vectorize_width(4)
  for (uint32_t i = 0; i < 4; i++) {
    w[i] = weights[i];
    uint32_t t0 = H + i_time - idelays[i];
    b1[i] = b2[i] = buf + H * indices[i] * width;
    b1[i] += ((t0 + 0) & Hm1) * width;
    b2[i] += ((t0 + 1) & Hm1) * width;
  }
}

// do one non-zero increment for cx1 & cx2
template <int width=8>
static void INLINE apply_ij(
  const cxb<width> &cx, const conn &c,
  const int nz, const int i_time, float *cx1, float *cx2)
{
  float *b1, *b2, w;
  prep_ij<width>(cx, c, i_time, nz, &b1, &b2, &w);
  inc<width>(cx1, b1, w);
  inc<width>(cx2, b2, w);
}  

// same but for 4 at once
template <int width=8>
static void INLINE apply4_ij(
  const cxb<width> &cx, const conn &c,
  const int nz, const int i_time, float *cx1, float *cx2)
{
  float *b1[4], *b2[4], w[4];
  prep4_ij<width>(cx, c, i_time, nz, b1, b2, w);
  inc<width>(cx1, b1[0], w[0]);
  inc<width>(cx2, b2[0], w[0]);
  inc<width>(cx1, b1[1], w[1]);
  inc<width>(cx2, b2[1], w[1]);
  inc<width>(cx1, b1[2], w[2]);
  inc<width>(cx2, b2[2], w[2]);
  inc<width>(cx1, b1[3], w[3]);
  inc<width>(cx2, b2[3], w[3]);
}

template <int width=8>
static void INLINE apply_all_node(
  const cxb<width> &cx, const conn &c,
  int i_time, int i_node, float *cx1, float *cx2) {
  zero<width>(cx1);
  zero<width>(cx2);

  int i0 = c.indptr[i_node];
  int i1 = c.indptr[i_node + 1];
  int nnz = i1 - i0;
  int n4 = nnz / 4;
  for (int i_n4 = 0; i_n4 < n4; i_n4++)
    apply4_ij<width>(cx, c, i0 + i_n4 * 4, i_time, cx1, cx2);
  nnz -= (n4 * 4);
  for (int nz = i1 - nnz; nz < i1; nz++)
    apply_ij<width>(cx, c, nz, i_time, cx1, cx2);
}

template <int width=8>
static void INLINE cx_j_b(
  const cxb<width> &cx, const conn &c, uint32_t t) {
  float cx1[width], cx2[width];
  for (uint32_t i = 0; i < cx.num_node; i++) {
    apply_all_node<width>(cx, c, t, i, cx1, cx2);
    load<width>(cx.cx1 + i * width, cx1);
    load<width>(cx.cx2 + i * width, cx2);
  }
}

template <int width=8>
static void INLINE cx_j_bs(
  const cxbs<width> &cx, const conn &c, uint32_t t) {
#pragma omp parallel for
  for (uint32_t i = 0; i < cx.num_batch; i++)
    cx_j_b<width>(cx.batch(i), c, t);
}
/*
                                       8I           ,dPYb,          
                                       8I           IP'`Yb          
                                       8I           I8  8I          
                                       8I           I8  8'          
  ,ggg,,ggg,,ggg,     ,ggggg,    ,gggg,8I   ,ggg,   I8 dP    ,g,    
 ,8" "8P" "8P" "8,   dP"  "Y8gggdP"  "Y8I  i8" "8i  I8dP    ,8'8,   
 I8   8I   8I   8I  i8'    ,8I i8'    ,8I  I8, ,8I  I8P    ,8'  Yb  
,dP   8I   8I   Yb,,d8,   ,d8',d8,   ,d8b, `YbadP' ,d8b,_ ,8'_   8) 
8P'   8I   8I   `Y8P"Y8888P"  P"Y8888P"`Y8888P"Y8888P'"Y88P' "YY8P8P
*/
struct jr {
  static const uint32_t num_svar=6, num_parm=14, num_cvar=1;
  static constexpr const char *const parms = "A,B,a,b,v0,nu_max,r,J,a_1,a_2,a_3,a_4,mu,I", *const name="jr";
  // with width=8 & -O3 -mavx2 -fveclib=libmvec -ffast-math & __restrict inputs,
  // clang generates straight asm no jumps
  // gcc also good, but drop -fveclib=libmvec
  template <int width>
  INLINE static void
  dfun(float *__restrict dx, const float *__restrict x, const float *__restrict c, const float *__restrict p)
  {
    #pragma omp simd
    for (int i=0; i<width; i++) {
      float y0=x[i], y1=x[i+width], y2=x[i+2*width], y3=x[i+3*width], y4=x[i+4*width], y5=x[i+5*width];
      float A=p[i+0*width],B=p[i+1*width],a=p[i+2*width],b=p[i+3*width],v0=p[i+4*width],nu_max=p[i+5*width],r=p[i+6*width],J=p[i+7*width],
        a_1=p[i+8*width],a_2=p[i+9*width],a_3=p[i+10*width],a_4=p[i+11*width],mu=p[i+12*width],I=p[i+13*width];
      float sigm_y1_y2 = 2.0f * nu_max / (1.0f + expf(r * (v0 - (y1 - y2))));
      float sigm_y0_1  = 2.0f * nu_max / (1.0f + expf(r * (v0 - (a_1 * J * y0))));
      float sigm_y0_3  = 2.0f * nu_max / (1.0f + expf(r * (v0 - (a_3 * J * y0))));
      dx[i+0*width] = y3;
      dx[i+1*width] = y4;
      dx[i+2*width] = y5;
      dx[i+3*width] = A * a * sigm_y1_y2 - 2.0f * a * y3 - a *a* 2.f * y0;
      dx[i+4*width] = A * a * (mu + a_2 * J * sigm_y0_1 + c[i]) - 2.0f * a * y4 - a *a * y1;
      dx[i+5*width] = B * b * (a_4 * J * sigm_y0_3) - 2.0f * b * y5 - b *b * y2;
    }
  }
  // adhoc adjustments to state variables after heun stages, e.g. to enforce constraints
  template <int width> INLINE static void adhoc(float *) { }
};

struct mpr {
  static const uint32_t num_svar=2, num_parm=6, num_cvar=1;
  static constexpr const char * const parms = "tau I Delta J eta cr", * const name = "mpr";
  static constexpr const float default_parms[6] = {1.0, 0.0, 1.0, 15.0, -5.0, 1.0};
  template <int width>
  INLINE static void
  dfun(float *__restrict dx, const float *__restrict x, const float *__restrict c, const float *__restrict p)
  {
    #pragma omp simd
    for (int i=0; i<width; i++) {
      float r=x[i+0*width],V=x[i+1*width];
      float tau=p[i+0*width],I=p[i+1*width],Delta=p[i+2*width],J=p[i+3*width],eta=p[i+4*width],cr=p[i+5*width];
      r = r * (r > 0);
      dx[i+0*width] = (1 / tau) * (Delta / (M_PI_F * tau) + 2.0f * r * V);
      dx[i+1*width] = (1 / tau) * (V * V + eta + J * tau * r + I + cr * c[i] - (M_PI_F * M_PI_F) * (r * r) * (tau * tau));
    }
  }
  template <int width> INLINE static void adhoc(float *x) {
    #pragma omp simd
    for (int i=0; i<width; i++) {
      x[i] = x[i] * (x[i] > 0);
    }
  }
};
/*
                      I8                                                I8           
                      I8                                                I8           
  gg               88888888                                          88888888        
  ""                  I8                                                I8           
  gg    ,ggg,,ggg,    I8    ,ggg,     ,gggg,gg   ,gggggg,    ,gggg,gg   I8    ,ggg,  
  88   ,8" "8P" "8,   I8   i8" "8i   dP"  "Y8I   dP""""8I   dP"  "Y8I   I8   i8" "8i 
  88   I8   8I   8I  ,I8,  I8, ,8I  i8'    ,8I  ,8'    8I  i8'    ,8I  ,I8,  I8, ,8I 
_,88,_,dP   8I   Yb,,d88b, `YbadP' ,d8,   ,d8I ,dP     Y8,,d8,   ,d8b,,d88b, `YbadP' 
8P""Y88P'   8I   `Y88P""Y8888P"Y888P"Y8888P"8888P      `Y8P"Y8888P"`Y88P""Y8888P"Y888
                                          ,d8I'                                      
                                        ,dP'8I                                       
                                       ,8"  8I                                       
                                       I8   8I                                       
                                       `8, ,8I                                       
                                        `Y8P"
*/
// steps a model for single batch of nodes size width assuming
// precomputed cx1 & cx2, and updates buffer in cx
template <typename model, int width=8>
static void heun_step(
  const cxb<width> &cx, float * __restrict states, 
  const float*  __restrict cx1, const float*  __restrict cx2, const float * __restrict params,
  const uint32_t i_node, const uint32_t i_time, const float dt
)
{
  constexpr uint8_t nsvar = model::num_svar;
  const uint32_t num_node = cx.num_node, horizon = cx.num_time;
  float x[nsvar*width], xi[nsvar*width]={}, dx1[nsvar*width]={}, dx2[nsvar*width]={};

  // load states
#pragma clang loop unroll(full)
  for (int svar=0; svar < nsvar; svar++) {
    load<width>(x+svar*width, states+width*(i_node + num_node*svar));
    zero<width>(xi+svar*width);
    zero<width>(dx1+svar*width);
    zero<width>(dx2+svar*width);
  }

  // Heun stage 1
  model::template dfun<width>(dx1, x, cx1, params);
#pragma clang loop unroll(full)
  for (int svar=0; svar < nsvar; svar++)
      heunpred<width>(x+svar*width, xi+svar*width, dx1+svar*width, dt);
  model::template adhoc<width>(xi);

  // Heun stage 2
  model::template dfun<width>(dx2, xi, cx2, params);
#pragma clang loop unroll(full)
  for (int svar=0; svar < nsvar; svar++)
      heuncorr<width>(x+svar*width, dx1+svar*width, dx2+svar*width, dt);
  model::template adhoc<width>(x);
#pragma clang loop unroll(full)
  for (int svar=0; svar < nsvar; svar++)
      load<width>(states+width*(i_node + num_node*svar), x+svar*width);

  // update buffer
  // TODO move out, to handle multiple cvars/cx/conns
  int write_time = i_time & (cx.num_time - 1);
  load<width>(cx.buf + width * (i_node * horizon + write_time),x);
}

template <typename model, int width=8>
static void step_batch(
  const cxb<width> &cx, const conn &c,
  float *x, // (num_svar, num_node, width)
  float *y, // (num_svar, num_node, width)
  // TODO try x layout as (num_node, num_svar, width)
  const float *p, // (num_node, num_parm, width)
  const bool p_varies_node,
  const uint32_t t0, const uint32_t nt, const float dt)
{
  float cx1[width], cx2[width];
  for (uint32_t i=0; i<(model::num_svar * cx.num_node); i++)
    zero<width>(y+i*width);
  for (uint32_t t=t0; t<(t0+nt); t++) {
    for (uint32_t i = 0; i < cx.num_node; i++) {
      apply_all_node<width>(cx, c, t, i, cx1, cx2);
      heun_step<model, width>(cx, x, cx1, cx2, 
        p_varies_node ? p+i*model::num_parm*width : p,
        i, t, dt);
    }
    for (uint32_t i=0; i<(model::num_svar * cx.num_node); i++)
      inc<width>(y+i*width, x+i*width, 1.0f);
  }
  for (uint32_t i=0; i<(model::num_svar * cx.num_node); i++)
    muls<width>(y+i*width, 1./nt);
}

template <typename model, int width=8>
static void step_batches(
  const cxbs<width> &cx, const conn &c,
  float *x, // (num_batch, num_svar, num_node, width)
  float *y, // (num_batch, num_svar, num_node, width)
  // TODO try x layout as (num_node, num_svar, width)
  const float *p, 
  const bool p_varies_node,
  const uint32_t t0, const uint32_t nt, const float dt)
{
// use omp if available or emscripten since std::thread causes emscripten to fail
#if _OPENMP || __EMSCRIPTEN__
  #pragma omp parallel for
#else
  std::vector<std::thread> threads;
#endif
  for (int b=0; b<cx.num_batch; b++) {
    const float *pb;
    float *xb=x + b * model::num_svar * cx.num_node * width;
    float *yb=y + b * model::num_svar * cx.num_node * width;
    // when p varies per node, shape is // (num_batch, num_node, num_parm, width)
    if (p_varies_node)
      pb = p + b * cx.num_node * model::num_parm * width;
    // otherwise p varies only per batch & item, (num_batch, num_parm, width)
    else
      pb = p + b * model::num_parm * width;
#if _OPENMP || __EMSCRIPTEN__
    step_batch<model, width>(cx.batch(b), c, xb, yb, pb, p_varies_node, t0, nt, dt);
#else
    threads.emplace_back(
      step_batch<model, width>, cx.batch(b), c, xb, yb, pb, p_varies_node, t0, nt, dt);
#endif
  }

#if _OPENMP || __EMSCRIPTEN__
#else
  for (auto &th : threads) th.join();
#endif
}

} // namespace tvbk
