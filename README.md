# Tuning-of-CUDA-Programs-by-Data-Layout-Technique
Tuning of CUDA Programs by Data Layout Technique for MMC
# CUDA Program Optimization for Chained Matrix Multiplication

This open-source repository focuses on optimizing chained matrix multiplication (CMM) algorithms using CUDA. The project explores both a baseline CUDA implementation and a tuned version using the Data Layout technique for improved performance.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [Compilation](#compilation)
  - [Running Experiments](#running-experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the domain of high-performance computing, optimizing GPU programs is crucial. This repository presents a CUDA implementation of the chained matrix multiplication algorithm, exploring parallelization and optimization techniques to enhance performance.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler with CUDA support
- Git (optional, for cloning the repository)

## Getting Started

### Compilation

1. Clone the repository:

    ```bash
    git clone https://github.com/neda-seifi/Tuning-of-CUDA-Programs-by-Data-Layout-Technique
    ```

2. Navigate to the project directory:

    ```bash
    cd Tuning-of-CUDA-Programs-by-Data-Layout-Technique
    ```

3. Compile the code:

    ```bash
    nvcc -o cmm_cuda cmm_cuda.cu
    nvcc -o cmm_cuda_layout cmm_cuda_with_data_layout.cu
    ```

### Running Experiments

1. Execute the serial version and CUDA baseline:

    ```bash
    ./cmm_serial
    ./cmm_cuda
    ```

2. Execute the tuned version with Data Layout technique:

    ```bash
    ./cmm_cuda_layout
    ```

## Results

The results of the experiments, including execution times for different matrix sizes, are stored in the `output.txt` file. Refer to this file to analyze the performance improvements achieved with the Data Layout technique.

## Contributing

Feel free to contribute by opening issues, submitting pull requests, or providing feedback. Contributions are welcome and appreciated.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
