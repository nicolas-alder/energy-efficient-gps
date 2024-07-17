# Energy-Efficient Gaussian Processes using Low-Precision Arithmetic

This repository accompanies the paper "Energy-Efficient Gaussian Processes using Low-Precision Arithmetic," published at ICML 2024. If you use any part of this code, please cite the following paper.

## Citation
Please cite our paper as follows:
```bash
@inproceedings{alderenergy,
  title={Energy-Efficient Gaussian Processes Using Low-Precision Arithmetic},
  author={Alder, Nicolas and Herbrich, Ralf},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

## Repository Structure
The repository is organized into the following main directories:

- [Software Benchmark](#software-benchmark)
- [FPGA Benchmarks](#fpga-benchmarks)

## Software Benchmark
This directory contains the software benchmark code for tracking operations and evaluating performances for the experiments described in the paper.

### Usage
1. **Execute Experiments**: 
   - Run `main.py` to perform experiments in parallel. 
   - Modify `main.py` to set the number of cores to use.

2. **Configuration**:
   - Edit the method `load_config()` in `src/FlexGP/utils.py` to specify the number of exponent bits, precision bits, datasets, and all other relevant experiment settings.

3. **Results**:
   - Result CSV files are written to the subfolder `experiments`.

4. **Dependencies**:
   - Use `requirements.txt` to install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

## FPGA Benchmarks
This directory contains subfolders for addition and multiplication benchmarks, sorted by respective bit sizes.

### Contents
- **Experiment Setup**: 
  - XSA files suitable for flashing onto the Genesys2 FPGA Board by Digilent.
  
- **Resource Utilization**: 
  - Images of FPGA core resource utilization.
  
- **Power Measurements**:
  - CSV files of power measurements.
  - Visualizations of the power measurements.

## Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/energy-efficient-gp.git
   cd energy-efficient-gp
  ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
  ```

3. **Run Software Benchmarks**:
   ```bash
   python main.py
   ```

4 . **Configure Experiments**:
Modify the load_config() function in src/FlexGP/utils.py as needed.

**License**
This project is licensed under the CC BY-NC 4.0 DEED License. See the [LICENSE](https://creativecommons.org/licenses/by-nc/4.0/) file for details.
