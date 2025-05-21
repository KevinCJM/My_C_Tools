// cal_max_dd_threadpool.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include <thread>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace py = pybind11;
const double  NAN_D   = std::numeric_limits<double>::quiet_NaN();
const int64_t BIG_INT = 1'000'000LL;

/* ---------- 工具：ns → "YYYYMMDD" ---------- */
inline std::string ns_to_yyyymmdd(int64_t ns)
{
    std::time_t sec = ns / 1'000'000'000LL;
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &sec);
#else
    gmtime_r(&sec, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d");
    return oss.str();
}

/* ---------- 主入口 ---------- */
py::tuple cal_max_draw_down(py::array_t<double> funds_val,
                            py::array_t<int64_t> day_arr)          // datetime64[ns] 已转 int64
{
    /* ---- 0. 解析 ---- */
    auto vbuf = funds_val.request();
    auto dbuf = day_arr.request();
    const ssize_t D = vbuf.shape[0];
    const ssize_t N = vbuf.shape[1];
    const double*  vptr = static_cast<double*>(vbuf.ptr);
    const int64_t* dptr = static_cast<int64_t*>(dbuf.ptr);

    /* ---- 1. 结果容器 ---- */
    std::vector<double>           max_dd (N, NAN_D);
    std::vector<int64_t>          rec_day(N, BIG_INT);
    std::vector<std::string>      dd_date(N, "");

    /* ---- 2. 线程池：按列静态分片 ---- */
    size_t n_th = std::min<std::size_t>(N, std::thread::hardware_concurrency());
    if (n_th == 0) n_th = 1;
    std::vector<std::thread> ths(n_th);

    for (size_t tid = 0; tid < n_th; ++tid)
        ths[tid] = std::thread([&, tid]{
            std::vector<double> cum(D);        // 本列临时缓冲
            std::vector<double> run_max(D);
            std::vector<double> dd(D);

            for (size_t c = tid; c < static_cast<size_t>(N); c += n_th)
            {
                /* ---- 2.1 nancumprod + leading NaN -- bfill ---- */
                double acc = 1.0;
                bool started = false;
                for (ssize_t r = 0; r < D; ++r) {
                    double ret = vptr[r*N + c];
                    if (std::isnan(ret)) {
                        cum[r] = NAN_D;                         // 暂留 NaN
                    } else {
                        if (!started) {
                            acc = 1.0;
                            started = true;
                        }
                        acc *= (1.0 + ret);                     // treat NaN as 0 return
                        cum[r] = acc;
                    }
                }
                /* backward fill (pandas.bfill) */
                double last = NAN_D;
                for (ssize_t r = D-1; r >= 0; --r) {
                    if (std::isnan(cum[r])) cum[r] = last;
                    else                    last    = cum[r];
                }

                if (std::isnan(cum[0])) {       // 全是 NaN
                    max_dd[c]  = 0.0;
                    rec_day[c] = BIG_INT;
                    dd_date[c] = ns_to_yyyymmdd(dptr[0]);
                    continue;
                }

                /* ---- 2.2 running max & drawdown ---- */
                double cur_max = cum[0];
                double min_dd  = 0.0;
                ssize_t min_idx = 0;
                for (ssize_t r = 0; r < D; ++r) {
                    cur_max = std::max(cur_max, cum[r]);
                    run_max[r] = cur_max;
                    dd[r] = (cum[r] - cur_max) / cur_max;      // ≤0
                    if (dd[r] < min_dd) { min_dd = dd[r]; min_idx = r; }
                }
                max_dd[c]  = std::abs(min_dd);
                dd_date[c] = ns_to_yyyymmdd(dptr[min_idx]);

                /* ---- 2.3 恢复天数 ---- */
                ssize_t rec = BIG_INT;
                for (ssize_t r = min_idx + 1; r < D; ++r)
                    if (cum[r] >= run_max[r]) { rec = r - min_idx; break; }
                if (min_idx == 0) rec = 0;      // 无回撤
                rec_day[c] = rec;
            }
        });

    for (auto& t : ths) t.join();

    /* ---- 3. 打包返回 ---- */
    py::array_t<double>  dd_arr (N, max_dd.data());
    py::array_t<int64_t> rt_arr (N, rec_day.data());
    py::list date_list;
    for (auto& s : dd_date) date_list.append(py::str(s));

    return py::make_tuple(dd_arr, date_list, rt_arr);
}

/* ---------- Pybind11 导出 ---------- */
PYBIND11_MODULE(cal_max_dd, m)
{
    m.def("cal_max_dd", &cal_max_draw_down,
          "Compute max drawdown, max drawdown date, and recovery days for each fund (thread-pool parallel)");
}
