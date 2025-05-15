from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'my_ctools.core',
        ['col_std_threadpool.cpp'],  # 路径修复
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3'],
    ),
]

setup(
    name='my-ctools',
    version='0.1.0',
    author='KevinCJM',
    description='Fast column-wise std calculator using C++ thread pool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=['my_ctools'],
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.6',
)
