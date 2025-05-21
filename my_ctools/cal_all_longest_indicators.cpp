// cal_longest_one_cpp.cpp  ---------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <thread>
#include <limits>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace py = pybind11;
const double NaN = std::numeric_limits<double>::quiet_NaN();

/* ns → "YYYYMMDD" --------------------------------------------------------- */
static std::string ns2ymd(int64_t ns) {
    std::time_t s = ns / 1'000'000'000LL;
    std::tm tm{}; gmtime_r(&s, &tm);
    std::ostringstream os; os << std::put_time(&tm, "%Y%m%d");
    return os.str();
}

/* NaN-safe cumprod --------------------------------------------------------- */
static void nancumprod(const double* col, ssize_t R, ssize_t stride,
                       bool invert, std::vector<double>& out) {
    out.resize(R);
    double acc = 1.0;
    for (ssize_t r = 0; r < R; ++r) {
        double v = col[r * stride];
        if (!std::isnan(v)) {
            if (invert) v = -v;
            acc *= (1.0 + v);         // NaN ⇒ *1
        }
        out[r] = acc;
    }
}

/* ────────────────────────────────────────────────────────────────────────── */
struct ColRes { double r; std::string s, e; long days; };

static ColRes calc_col(const double* col,
                       const int64_t* dates,
                       ssize_t R, ssize_t C,
                       bool positive_mode) {
    /* 0. 复制并把内部 NaN → 0.0 --------------------------------------- */
    std::vector<double> v(R);
    for (ssize_t r = 0; r < R; ++r) v[r] = col[r * C];

    ssize_t first = -1, last = -1;
    for (ssize_t r = 0; r < R && first == -1; ++r)
        if (!std::isnan(v[r])) first = r;
    for (ssize_t r = R - 1; r >= 0 && last == -1; --r)
        if (!std::isnan(v[r])) last = r;
    if (first != -1 && last != -1)
        for (ssize_t r = first + 1; r < last; ++r)
            if (std::isnan(v[r])) v[r] = 0.0;

    /* 1. 找“最后一个最长正收益段” ------------------------------------ */
    long best_len = 0, cur = 0;
    ssize_t best_end = -1;                // 记录段尾行号
    for (ssize_t r = 0; r < R; ++r) {
        bool pos = (!std::isnan(v[r]) && v[r] > 0);
        if (pos) {
            ++cur;
            if (cur >= best_len) {        // >= 保留最后一个
                best_len = cur;
                best_end = r;
            }
        } else {
            cur = 0;
        }
    }
    if (best_len == 0)                    // 没有正段
        return {0.0, "nan", "nan", 0};

    ssize_t start = best_end - best_len + 1;
    ssize_t end   = best_end;

    /* 2. 累计净值 & 区间收益 ---------------------------------------- */
    std::vector<double> nv;
    nancumprod(col, R, C, !positive_mode, nv);

    ssize_t row_before = (start == 0 ? R - 1 : start - 1);
    double first_nv = nv[row_before];
    double last_nv  = nv[end];
    double ret      = (first_nv != 0.0) ? (last_nv / first_nv - 1.0) : NaN;

    return {ret, ns2ymd(dates[start]), ns2ymd(dates[end]), best_len};
}

/* ======================================================================= */
py::tuple main_cal_longest_one_for_all(py::array_t<double>  a_value,
                                       py::array_t<int64_t> dates,
                                       std::string i_code = "positive")
{
    auto vbuf = a_value.request();
    auto dbuf = dates.request();
    const ssize_t R = vbuf.shape[0], C = vbuf.shape[1];
    const double*  vptr = static_cast<double*>(vbuf.ptr);
    const int64_t* dptr = static_cast<int64_t*>(dbuf.ptr);

    std::vector<double>  r(C, 0.0);
    std::vector<std::string> s(C, "nan"), e(C, "nan");
    std::vector<long>    days(C, 0);

    bool positive_mode = (i_code == "positive");

    /* 多线程处理列 ---------------------------------------------------- */
    std::size_t n_th = std::min<std::size_t>(C, std::thread::hardware_concurrency());
    if (!n_th) n_th = 1;
    std::vector<std::thread> ths(n_th);

    for (std::size_t tid = 0; tid < n_th; ++tid)
        ths[tid] = std::thread([&, tid]() {
            for (ssize_t c = tid; c < C; c += n_th) {
                ColRes res = calc_col(vptr + c, dptr, R, C, positive_mode);
                r[c]     = res.r;
                s[c]     = res.s;
                e[c]     = res.e;
                days[c]  = res.days;
            }
        });
    for (auto& t : ths) t.join();

    /* 打包返回 -------------------------------------------------------- */
    auto to_np_dbl = [&](std::vector<double>& v){ return py::array_t<double>(C, v.data()); };
    auto to_np_long= [&](std::vector<long>&   v){ return py::array_t<long  >(C, v.data()); };

    py::list sl, el;
    for (auto& x:s) sl.append(py::str(x));
    for (auto& x:e) el.append(py::str(x));

    return py::make_tuple(to_np_dbl(r), sl, el, to_np_long(days));

}

/* --------------------------------------------------------------------- */
PYBIND11_MODULE(cal_all_longest_indicators, m)
{
    m.def("cal_all_longest_indicators",
          &main_cal_longest_one_for_all,
          py::arg("a_value"), py::arg("dates"),
          py::arg("i_code") = "positive",
          "C++ translation of Python main_cal_longest_one_for_all");
}
