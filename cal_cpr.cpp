// cal_cpr.cpp  ----  线程池(静态分片)并行版
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thread>

namespace py = pybind11;
const double nan_val = std::numeric_limits<double>::quiet_NaN();

/* ------------------- 主函数 ------------------- */
py::array_t<double> cal_cpr(py::array_t<int> f_type,
                            py::array_t<double> funds_value)
{
    /* ---------- 0. 解析输入 ---------- */
    auto t_buf = f_type.request();
    auto v_buf = funds_value.request();

    const int*    type_ptr  = static_cast<int*>(t_buf.ptr);
    const double* value_ptr = static_cast<double*>(v_buf.ptr);
    const ssize_t D = v_buf.shape[0];      // 天
    const ssize_t F = v_buf.shape[1];      // 基金

    /* ---------- 1. fund_type -> 列索引 ---------- */
    std::unordered_map<int, std::vector<size_t>> type2cols;
    type2cols.reserve(F);
    for (size_t c = 0; c < F; ++c) type2cols[type_ptr[c]].push_back(c);

    /* ---------- 2. 计算每日中位数 (单线程足够) ---------- */
    std::vector<std::vector<double>> med(D, std::vector<double>(F, nan_val));

    for (const auto& [ftype, cols] : type2cols)
        for (ssize_t d = 0; d < D; ++d) {
            std::vector<double> buf;
            buf.reserve(cols.size());
            for (size_t c : cols) {
                double v = value_ptr[d * F + c];
                if (!std::isnan(v)) buf.push_back(v);
            }
            if (buf.empty()) continue;
            std::nth_element(buf.begin(), buf.begin() + buf.size() / 2, buf.end());
            double m = (buf.size() & 1)
                         ? buf[buf.size() / 2]
                         : 0.5 * ( *std::max_element(buf.begin(), buf.begin() + buf.size() / 2)
                                 + *std::min_element(buf.begin() + buf.size() / 2, buf.end()));
            for (size_t c : cols) med[d][c] = m;
        }

    /* ---------- 3. 构造比较矩阵 cmp (0/1/-1) ---------- */
    std::vector<std::vector<int8_t>> cmp(D, std::vector<int8_t>(F, -1));
    for (ssize_t d = 0; d < D; ++d)
        for (ssize_t c = 0; c < F; ++c) {
            double v = value_ptr[d * F + c];
            double m = med[d][c];
            cmp[d][c] = (std::isnan(v) || std::isnan(m)) ? -1 : (v >= m);
        }

    /* ---------- 4. 并行逐基金计算 CPR ---------- */
    std::vector<double> cpr_list(F, nan_val);
    size_t n_threads = std::min<size_t>(F, std::thread::hardware_concurrency());
    if (n_threads == 0) n_threads = 1;                 // Fallback

    std::vector<std::thread> threads(n_threads);

    for (size_t t = 0; t < n_threads; ++t) {
        threads[t] = std::thread([&, t](){
            for (size_t col = t; col < static_cast<size_t>(F); col += n_threads) {
                int ww = 0, ll = 0, wl = 0, lw = 0;
                bool have_last = false;
                int8_t last = 0;

                for (ssize_t d = 0; d < D; ++d) {
                    int8_t cur = cmp[d][col];
                    if (cur == -1) continue;          // 跳过 NaN，不重置 last
                    if (!have_last) {                 // 第一次有效
                        last = cur;
                        have_last = true;
                        continue;
                    }
                    if ( last == 1 && cur == 1) ww++;
                    else if(last == 0 && cur == 0) ll++;
                    else if(last == 1 && cur == 0) wl++;
                    else if(last == 0 && cur == 1) lw++;
                    last = cur;
                }
                int denom = wl + lw;
                if (denom > 0) cpr_list[col] = double(ww + ll) / denom;
                /* denom==0 时保持 NaN */
            }
        });
    }
    for (auto& th : threads) th.join();

    /* ---------- 5. 返回 NumPy 数组 ---------- */
    return py::array_t<double>(F, cpr_list.data());
}

/* ------------------- 模块导出 ------------------- */
PYBIND11_MODULE(cal_cpr, m) {
    m.def("cal_cpr", &cal_cpr,
          "Compute CPR for each fund (thread-pool parallel)");
}
