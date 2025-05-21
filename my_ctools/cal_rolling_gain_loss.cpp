// cal_rolling_gain.cpp  ------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <thread>
#include <regex>
#include <limits>
#include <ctime>
#include <algorithm>

namespace py = pybind11;
const double NaN = std::numeric_limits<double>::quiet_NaN();

/* ---------- ns 时间戳 与 tm 互转 ---------- */
static void ns_to_tm(int64_t ns, std::tm& out){
    std::time_t s = ns / 1'000'000'000LL;
#if defined(_WIN32)
    gmtime_s(&out,&s);
#else
    gmtime_r(&s,&out);
#endif
}
static int64_t tm_to_ns(const std::tm& in){
#if defined(_WIN32)
    std::tm t = in;
    std::time_t s = _mkgmtime(&t);
#else
    std::time_t s = timegm(const_cast<std::tm*>(&in));
#endif
    return int64_t(s)*1'000'000'000LL;
}

/* ---------- 加 N 月 ---------- */
static int64_t add_months_ns(int64_t ns,int n_months){
    std::tm tm{}; ns_to_tm(ns,tm);
    int y=tm.tm_year+1900, m=tm.tm_mon+1, d=tm.tm_mday;
    int tot=y*12+(m-1)+n_months;
    int ny=tot/12, nm=tot%12+1;
    static const int md[12]={31,28,31,30,31,30,31,31,30,31,30,31};
    int maxd=md[nm-1];
    if(nm==2 && ((ny%4==0&&ny%100!=0)||ny%400==0)) maxd=29;
    if(d>maxd) d=maxd;
    std::tm ntm{}; ntm.tm_year=ny-1900; ntm.tm_mon=nm-1; ntm.tm_mday=d;
    ntm.tm_hour=tm.tm_hour; ntm.tm_min=tm.tm_min; ntm.tm_sec=tm.tm_sec;
    return tm_to_ns(ntm);
}

/* ---------- NaN 安全的均值/中位数 ---------- */
static double nan_mean(const std::vector<double>& v){
    double s=0; size_t n=0;
    for(double x:v) if(!std::isnan(x)){ s+=x; ++n; }
    return n? s/n : NaN;
}
static double nan_median(std::vector<double> v){
    v.erase(std::remove_if(v.begin(),v.end(),[](double x){return std::isnan(x);}),v.end());
    if(v.empty()) return NaN;
    size_t k=v.size()/2;
    std::nth_element(v.begin(),v.begin()+k,v.end());
    double med=v[k];
    if(!(v.size()&1)){
        auto it=std::max_element(v.begin(),v.begin()+k);
        med=0.5*(med+*it);
    }
    return med;
}

/* ======================================================================= */
py::tuple cal_rolling_gain(std::string           i_code,
                           py::array_t<double>   funds_val,   // (D,N)
                           py::array_t<int64_t>  start_idx,   // (N,)
                           py::array_t<int64_t>  end_idx,     // (N,)
                           py::array_t<int64_t>  day_arr)     // (D,) datetime64[ns]
{
    auto vbuf = funds_val.request();
    auto sbuf = start_idx.request();
    auto ebuf = end_idx.request();
    auto dbuf = day_arr.request();

    const ssize_t D = vbuf.shape[0], N = vbuf.shape[1];
    const double*  vptr = static_cast<double*>(vbuf.ptr);
    const int64_t* sptr = static_cast<int64_t*>(sbuf.ptr);
    const int64_t* eptr = static_cast<int64_t*>(ebuf.ptr);
    const int64_t* dptr = static_cast<int64_t*>(dbuf.ptr);

    /* 1) 解析周期  -------------------------------------------------------- */
    std::smatch mm;
    if(!std::regex_search(i_code,mm,std::regex(R"((\d+)([MY]))")))
        throw std::runtime_error("i_code 格式错误");
    int  num  = std::stoi(mm[1]);
    char unit = mm[2].str()[0];
    int months = (unit=='M')? num : num*12;     // 仅需 1M/3M/1Y 等

    /* 2) 预计算未来索引 month_later_i ------------------------------------- */
    std::vector<ssize_t> fut(D,D-1);
    ssize_t cur=0;
    for(ssize_t i=0;i<D;++i){
        int64_t tgt = add_months_ns(dptr[i],months);
        while(cur<D && dptr[cur] < tgt) ++cur;
        fut[i]=(cur<D)?cur:D-1;
    }
    size_t tail = std::count(fut.begin(),fut.end(), D-1);

    /* 3) 输出容器初始化 ---------------------------------------------------- */
    auto vec=[&](){return std::vector<double>(N,NaN);};
    std::vector<double> avg=vec(), med=vec(), win=vec(),
                        w0=vec(), w5=vec(), w10=vec(),
                        l0=vec(), l5=vec(), l10=vec();

    /* 4) 多线程并行按列处理 ---------------------------------------------- */
    size_t n_th = std::min<std::size_t>(N, std::thread::hardware_concurrency());
    if(!n_th) n_th=1;
    std::vector<std::thread> th(n_th);

    for(size_t tid=0; tid<n_th; ++tid)
        th[tid]=std::thread([&,tid]{
            std::vector<double> nv(D), ret(D);
            for(size_t c=tid;c<static_cast<size_t>(N);c+=n_th)
            {
                /* a) 计算虚拟净值 with NaN mask */
                double acc=1.0;
                bool   started=false;
                for(ssize_t r=0;r<D;++r){
                    if(r<sptr[c] || r>eptr[c])       { nv[r]=NaN; continue; }
                    double v=vptr[r*N+c];
                    if(std::isnan(v)){               // NaN 看作 0%：输出 acc
                        nv[r]=acc;
                        continue;
                    }
                    if(!started){ acc=1.0; started=true; }
                    acc*=1.0+v;
                    nv[r]=acc;
                }

                /* b) N 月收益率 */
                for(ssize_t r=0;r<D;++r){
                    ssize_t j=fut[r];
                    if(std::isnan(nv[r])||std::isnan(nv[j])) { ret[r]=NaN; continue; }
                    ret[r]=nv[j]/nv[r]-1.0;
                }
                /* c) 将最后 tail+1 行置 NaN */
                for(ssize_t r=D-static_cast<ssize_t>(tail)-1; r<D; ++r) ret[r]=NaN;

                /* d) 统计各项指标 */
                std::size_t notnan=0,pos=0, w0c=0,w5c=0,w10c=0,l0c=0,l5c=0,l10c=0;
                for(double x:ret){
                    if(std::isnan(x)) continue;
                    ++notnan;
                    if(x>0) ++pos;
                    if( 0.0< x && x<=0.05) ++w0c;
                    else if(0.05< x && x<=0.10) ++w5c;
                    else if(x>0.10) ++w10c;
                    else if(-0.05 <= x && x<=0.0) ++l0c;
                    else if(-0.10 <= x && x<-0.05) ++l5c;
                    else if(x<-0.10) ++l10c;
                }
                std::vector<double> vec_ret(ret.begin(),ret.end());
                avg[c]=nan_mean(vec_ret);
                med[c]=nan_median(vec_ret);
                if(notnan){
                    win[c]=double(pos)/notnan;
                    w0[c]=double(w0c)/notnan; w5[c]=double(w5c)/notnan; w10[c]=double(w10c)/notnan;
                    l0[c]=double(l0c)/notnan; l5[c]=double(l5c)/notnan; l10[c]=double(l10c)/notnan;
                }
            }
        });
    for(auto& t:th) t.join();

    /* 5) 转 NumPy 返回 ---------------------------------------------------- */
    auto to_np=[&](std::vector<double>& v){ return py::array_t<double>(N,v.data()); };

    return py::make_tuple( to_np(avg), to_np(med), to_np(win),
                           to_np(w0),  to_np(w5),  to_np(w10),
                           to_np(l0),  to_np(l5),  to_np(l10) );
}

/* ------------------------------------------------------------------------- */
PYBIND11_MODULE(cal_rolling_gain_loss, m)
{
    m.def("cal_rolling_gain_loss", &cal_rolling_gain,
          "Fully numpy-consistent C++ implementation of cal_rolling_gain");
}
