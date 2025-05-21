// cal_largest_gain_threaded.cpp  — fully aligned with Python logic
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <limits>
#include <thread>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace py = pybind11;
const double NaN = std::numeric_limits<double>::quiet_NaN();

/* -------- utils -------- */
std::string ns2ymd(int64_t ns){
    std::time_t sec = ns/1'000'000'000LL;
    std::tm tm{}; gmtime_r(&sec,&tm);
    std::ostringstream os; os<<std::put_time(&tm,"%Y%m%d");
    return os.str();
}

/* replace internal NaN with zero */
void replace_inner_nan(std::vector<double>& col){
    ssize_t n=col.size();
    ssize_t first=-1,last=-1;
    for(ssize_t i=0;i<n && first==-1;++i) if(!std::isnan(col[i])) first=i;
    for(ssize_t i=n-1;i>=0 && last==-1;--i) if(!std::isnan(col[i])) last=i;
    if(first==-1||last==-1) return;
    for(ssize_t i=first+1;i<last;++i) if(std::isnan(col[i])) col[i]=0.0;
}

/* nancumprod identical to numpy's: treat NaN as multiplier 1 (skip) */
void nancumprod(const std::vector<double>& arr,std::vector<double>& out,bool invert){
    ssize_t n=arr.size(); out.resize(n);
    double acc=1.0;
    for(ssize_t i=0;i<n;++i){
        double v=arr[i];
        if(!std::isnan(v)){
            if(invert) v=-v;
            acc*=1.0+v;
        }
        out[i]=acc;
    }
}

struct ColRes{double r; long p; std::string s; std::string l;};

ColRes process_col(const double* col_ptr,ssize_t rows,ssize_t stride,const int64_t* dates,bool positive){
    /* copy column */
    std::vector<double> col(rows); for(ssize_t r=0;r<rows;++r) col[r]=col_ptr[r*stride];
    replace_inner_nan(col);

    /* build selected array >0 else NaN */
    std::vector<double> sel(rows,NaN);
    for(ssize_t r=0;r<rows;++r) if(col[r]>0) sel[r]=col[r];

    /* cumulative product */
    std::vector<double> cp; nancumprod(sel,cp,!positive);

    /* build mf (mask of NaNs) */
    std::vector<double> denom(rows);
    double last=1.0;
    for(ssize_t r=0;r<rows;++r){
        if(std::isnan(sel[r])) { denom[r]=last; }
        else { denom[r]=NaN; last=cp[r]; }
    }
    /* forward fill denom */
    for(ssize_t r=0;r<rows;++r){
        if(std::isnan(denom[r])) denom[r]= (r? denom[r-1]:1.0);
    }

    /* ratio */
    std::vector<double> ratio(rows,NaN);
    for(ssize_t r=0;r<rows;++r){
        if(!std::isnan(sel[r])){
            ratio[r]=cp[r]/denom[r];
            if(ratio[r]==1.0) ratio[r]=NaN;
        }
    }

    /* find extreme */
    double extreme = positive? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    for(double v:ratio) if(!std::isnan(v)){
        if((positive && v>extreme)||(!positive && v<extreme)) extreme=v; }
    if(std::isinf(extreme)) return {0,0,"nan","nan"};

    /* last index of extreme */
    ssize_t last_idx=-1;
    for(ssize_t r=rows-1;r>=0;--r) if(!std::isnan(ratio[r]) && ratio[r]==extreme){ last_idx=r; break; }

    /* first_idx = first row after the last NaN before last_idx (contiguous block start) */
    ssize_t first_idx = 0;
    for (ssize_t r = last_idx; r >= 0; --r) {
        if (std::isnan(ratio[r])) { first_idx = r + 1; break; }
        if (r == 0) first_idx = 0;  // reached top without NaN
    }

    long period = static_cast<long>(last_idx - first_idx + 1);

    return {extreme-1.0, period, ns2ymd(dates[first_idx]), ns2ymd(dates[last_idx])};
}

py::dict main_cal_largest_one_for_all(py::array_t<double> array_value,
                                      py::array_t<int64_t> dates,
                                      std::string i_code="positive"){
    auto buf=array_value.request(); auto dbuf=dates.request();
    ssize_t R=buf.shape[0], C=buf.shape[1];
    const double* vptr=static_cast<double*>(buf.ptr);
    const int64_t* dptr=static_cast<int64_t*>(dbuf.ptr);

    std::vector<double> r(C,0.0); std::vector<long> p(C,0);
    std::vector<std::string> s(C,"nan"),l(C,"nan");
    bool positive=(i_code=="positive");

    size_t threads=std::min<size_t>(C,std::thread::hardware_concurrency());
    if(!threads) threads=1; std::vector<std::thread> th(threads);

    for(size_t t=0;t<threads;++t){
        th[t]=std::thread([&,t]{
            for(ssize_t c=t;c<C;c+=threads){
                ColRes res=process_col(vptr+c,R,C,dptr,positive);
                r[c]=res.r; p[c]=res.p; s[c]=res.s; l[c]=res.l;
            }
        });
    }
    for (auto& thd : th) thd.join();

    py::array_t<double> r_arr(C,r.data());
    py::array_t<long>   p_arr(C,p.data());
    py::list s_list,l_list; for(auto& x:s) s_list.append(py::str(x));
    for(auto& x:l) l_list.append(py::str(x));

    py::dict res; res["r"]=r_arr; res["p"]=p_arr; res["s"]=s_list; res["l"]=l_list;
    return res;
}

PYBIND11_MODULE(cal_all_largest_indicators,m){
    m.def("cal_all_largest_indicators",&main_cal_largest_one_for_all,
          py::arg("array_value"),py::arg("dates"),py::arg("i_code")="positive");
}
