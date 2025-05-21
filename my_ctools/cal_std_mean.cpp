#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <thread>

namespace py = pybind11;

void compute_column_std(double* ptr, size_t rows, size_t cols, size_t j, double* result) {
    double sum = 0.0, sum_sq = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < rows; ++i) {
        double val = ptr[i * cols + j];
        if (!std::isnan(val)) {
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }
    if (count > 1) {
        double mean = sum / count;
        result[j] = std::sqrt((sum_sq - count * mean * mean) / (count - 1));
    } else {
        result[j] = std::nan("");
    }
}

py::array_t<double> column_std_threadpool(py::array_t<double> input) {
    auto buf = input.request();
    size_t rows = buf.shape[0], cols = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<double> result(cols, std::nan(""));
    size_t n_threads = std::min(cols, (size_t)std::thread::hardware_concurrency());
    std::vector<std::thread> threads(n_threads);
    for (size_t t = 0; t < n_threads; ++t) {
        threads[t] = std::thread([=, &result]() {
            for (size_t j = t; j < cols; j += n_threads)
                compute_column_std(ptr, rows, cols, j, result.data());
        });
    }
    for (auto& t : threads) t.join();
    return py::array_t<double>(result.size(), result.data());
}

// simd_mean_std_threadpool.cpp – 使用 SIMD 指令集和 std::thread 线程池（无 OpenMP）进行列统计
// • x86 架构使用 AVX2 内建函数 | Apple Silicon (ARM) 使用 NEON 内建函数
// • 在计算均值（mean）和无偏标准差（std, ddof=1）时忽略 NaN 值
// • 单次线程池处理：每个线程处理部分列数据
//--------------------------------------------------------------
#include <pybind11/pybind11.h>     // Pybind11 核心头文件
#include <pybind11/numpy.h>        // 支持 NumPy 数组类型
#include <vector>                  // 使用 vector 容器
#include <thread>                  // 使用线程库
#include <cmath>                   // 使用数学函数
#include <limits>                  // 数值极限定义（如 NaN）
#include <algorithm>               // 使用 min 函数等

// ==== SIMD 架构选择 ====
#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>           // 使用 AVX2 指令集（每次处理 4 个 double）
  #define SIMD_WIDTH 4             // 每次处理 4 个 double 数据
#elif defined(__aarch64__)
  #include <arm_neon.h>            // 使用 NEON 指令集（每次处理 2 个 double）
  #define SIMD_WIDTH 2             // 每次处理 2 个 double 数据
#else
  #define SIMD_WIDTH 1             // 默认使用标量计算
#endif

namespace py = pybind11;
static constexpr double nanv = std::numeric_limits<double>::quiet_NaN();  // 定义 NaN 常量

//------------------------------------------------------------------
// 辅助结构体：用于保存单列的统计结果（均值和标准差）
//------------------------------------------------------------------
struct ColStat {
    double mean;   // 均值
    double std;    // 标准差
};

//------------------------------------------------------------------
// 函数：计算单列的均值和标准差（忽略 NaN 值）
//------------------------------------------------------------------
static ColStat column_stat(const double* col_ptr, std::size_t n_rows, std::size_t stride) {
    double sum = 0.0, sum_sq = 0.0;
    std::size_t cnt = 0;

    // ---- AVX2 路径（适用于 x86 架构）----
#if SIMD_WIDTH == 4
    __m256d vsum   = _mm256_setzero_pd();  // 初始化向量求和寄存器
    __m256d vsumSq = _mm256_setzero_pd();  // 初始化平方和寄存器
    std::size_t i = 0;

    for (; i + SIMD_WIDTH <= n_rows; i += SIMD_WIDTH) {
        // 加载 4 个元素到 SIMD 寄存器中
        __m256d v = _mm256_set_pd(col_ptr[(i+3)*stride], col_ptr[(i+2)*stride],
                                   col_ptr[(i+1)*stride], col_ptr[i*stride]);

        // 判断是否为 NaN（v == v 成立即非 NaN）
        __m256d mask = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
        __m256d vmasked = _mm256_and_pd(v, mask);  // 应用掩码过滤 NaN

        // 向量加法累加
        vsum   = _mm256_add_pd(vsum,   vmasked);
        vsumSq = _mm256_add_pd(vsumSq, _mm256_mul_pd(vmasked, vmasked));

        // 统计有效元素数量
        uint32_t m = _mm256_movemask_pd(mask);
        cnt += __builtin_popcount(m);
    }

    alignas(32) double buf[4];
    _mm256_store_pd(buf, vsum);   sum     += buf[0] + buf[1] + buf[2] + buf[3];
    _mm256_store_pd(buf, vsumSq); sum_sq  += buf[0] + buf[1] + buf[2] + buf[3];

    // ---- NEON 路径（适用于 ARM 架构）----
#elif SIMD_WIDTH == 2
    std::size_t i = 0;
    for (; i + 2 <= n_rows; i += 2) {
        float64x2_t v = {col_ptr[i*stride], col_ptr[(i+1)*stride]};
        uint64x2_t mask = vceqq_f64(v, v);  // 非 NaN 掩码

        if (vgetq_lane_u64(mask, 0)) {
            double val = vgetq_lane_f64(v, 0);
            sum += val; sum_sq += val * val; ++cnt;
        }
        if (vgetq_lane_u64(mask, 1)) {
            double val = vgetq_lane_f64(v, 1);
            sum += val; sum_sq += val * val; ++cnt;
        }
    }

    // ---- 标量路径（回退）----
#else
    std::size_t i = 0;
#endif

    // 处理剩余未被 SIMD 处理的数据（尾部）
    for (; i < n_rows; ++i) {
        double v = col_ptr[i*stride];
        if (!std::isnan(v)) {
            sum += v;
            sum_sq += v * v;
            ++cnt;
        }
    }

    // 如果有效样本数小于等于 1，则返回 NaN
    if (cnt <= 1) return {nanv, nanv};

    // 计算均值和方差
    double mean = sum / cnt;
    double var  = (sum_sq - cnt * mean * mean) / (cnt - 1);

    return {mean, std::sqrt(var)};  // 返回均值和标准差
}

//------------------------------------------------------------------
// Python 对接函数：多线程并行调用 column_stat 函数
//------------------------------------------------------------------
py::array_t<double> mean_std_simd_threadpool(py::array_t<double> input){
    auto buf = input.request();
    std::size_t rows = buf.shape[0];       // 行数
    std::size_t cols = buf.shape[1];       // 列数
    const double* base = static_cast<double*>(buf.ptr);  // 数据起始地址

    std::vector<double> means(cols), stds(cols);  // 存储每列的均值和标准差

    // 自动确定线程数（不超过 CPU 核心数）
    std::size_t n_thr = std::min<std::size_t>(cols, std::thread::hardware_concurrency());
    if (n_thr == 0) n_thr = 1;

    std::vector<std::thread> pool;
    pool.reserve(n_thr);

    // 创建线程池并分配任务
    for (std::size_t t = 0; t < n_thr; ++t){
        pool.emplace_back([=, &means, &stds](){
            for (std::size_t j = t; j < cols; j += n_thr){
                const double* col_ptr = base + j;  // 指向第 j 列第一个元素
                ColStat st = column_stat(col_ptr, rows, cols);
                means[j] = st.mean;
                stds[j] = st.std;
            }
        });
    }

    // 等待所有线程完成
    for (auto& th : pool) th.join();

    // 构造输出数组（2 行 × cols 列）
    std::vector<ssize_t> shape = {2, static_cast<ssize_t>(cols)};
    py::array_t<double> out(shape);
    auto r = out.mutable_unchecked<2>();

    // 填充输出数组
    for (std::size_t j = 0; j < cols; ++j) {
        r(0, j) = means[j];  // 第一行是均值
        r(1, j) = stds[j];   // 第二行是标准差
    }

    return out;
}

PYBIND11_MODULE(cal_std_mean, m) {
    m.def("cal_std_mean", &column_std_threadpool, "Compute column-wise std (ddof=1, ignore NaNs)");
    m.def("cal_std_mean_simd", &mean_std_simd_threadpool, "Compute column-wise std with SIMD (ddof=1, ignore NaNs)");
}
