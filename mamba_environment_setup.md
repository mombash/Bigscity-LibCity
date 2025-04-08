# Reconfiguring a Conda Environment for Mamba in LibCity Framework

This document outlines the process of rebuilding a corrupted conda environment for running the Mamba state space model in the LibCity urban computing framework.

## 1. Environment Assessment and Cleanup

First, we identified and removed the problematic environment:

```bash
# Check existing environments
conda env list

# Remove the corrupted environment
conda remove -n libcity-mamba --all -y
```

**Notes:** 
- Always check existing environments before deletion to ensure you're targeting the correct one
- The `--all` flag ensures complete removal of all packages in the environment
- The `-y` flag automatically confirms the removal process without prompts

## 2. Creating a Fresh Environment

```bash
conda create -n libcity-mamba python=3.9 -y
```

**Notes:**
- We chose Python 3.9 instead of 3.7 (mentioned in LibCity docs) because:
  - Mamba SSM library requires Python 3.8+ 
  - Python 3.9 provides better compatibility between modern ML libraries
  - Many newer packages no longer support Python 3.7

## 3. Setting Up PyTorch

```bash
conda activate libcity-mamba
conda install pytorch==2.0.0 torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
```

**Notes:**
- PyTorch 2.0.0 was selected because:
  - It's compatible with CUDA 11.8 
  - It works with Mamba SSM 1.2.2
  - Newer versions might have compatibility issues with Mamba SSM 1.2.2
  - We avoided PyTorch 1.7.1 (mentioned in LibCity docs) as it's too old for Mamba SSM
- Using CUDA 11.8 instead of system CUDA 12.0:
  - PyTorch binaries are precompiled for specific CUDA versions
  - This approach prevents CUDA version mismatch errors

## 4. Installing Mamba SSM and Core Dependencies

```bash
pip install mamba-ssm==1.2.2 tensorboardX scikit-learn pandas numpy matplotlib tqdm dgl==2.1.0 einops ninja
```

**Notes:**
- `mamba-ssm==1.2.2`: Specific version known to work with our PyTorch version
- `einops`: Required for tensor manipulation in Mamba models
- `ninja`: Accelerates C++ code compilation for Mamba's CUDA operations
- `dgl==2.1.0`: Initially installed, later downgraded to ensure compatibility 

## 5. Addressing NumPy Version Conflicts

```bash
pip install numpy==1.23.5
```

**Notes:**
- NumPy 1.23.5 was chosen as a compromise:
  - LibCity originally used NumPy 1.19.4 (too old for newer packages)
  - NumPy 2.x caused compatibility issues with older libraries
  - 1.23.5 is compatible with both modern packages and LibCity code
- NumPy version conflicts are common in ML stacks due to different requirements between packages

## 6. Installing Additional LibCity Dependencies

```bash
pip install tabulate hyperopt geopy fastdtw
```

**Notes:**
- These are core dependencies for LibCity data processing
- We installed them incrementally to better identify and resolve dependency conflicts
- Older packages like `ray[tune]==1.2.0` specified in the LibCity requirements were unavailable, so we used newer versions

## 7. Installing Missing Dependencies

As we encountered errors, we installed additional required packages:

```bash
# For graph data processing in models
pip install infomap

# For wavelet transforms in STWave model
pip install PyWavelets

# For geometric deep learning
pip install torch-geometric

# For visualization and logging
pip install tensorboard
```

**Notes:**
- `torch-geometric`: Needed for graph-based models and operations in traffic networks
- `infomap`: Used for community detection in network analysis
- These dependencies weren't initially obvious but were required by various models in the LibCity framework

## 8. Fixing DGL Compatibility Issues

```bash
pip uninstall -y dgl
pip install dgl==0.6.1
```

**Notes:**
- DGL (Deep Graph Library) version 0.6.1 was specifically needed:
  - Newer versions (2.1.0) had incompatible dependencies with other packages
  - LibCity was originally designed with this older version
  - This demonstrates the importance of matching library versions to the codebase

## Key Insights and Best Practices

1. **Version Compatibility**: Carefully balance between:
   - Framework requirements (LibCity)
   - Model requirements (Mamba SSM) 
   - System constraints (CUDA version)

2. **Incremental Installation**: 
   - Install packages incrementally to isolate dependency issues
   - Start with core dependencies, then add specific model requirements

3. **Python Version Selection**:
   - Choose a Python version that supports all required libraries
   - Sometimes compromising between the framework's recommended version and what modern libraries require

4. **Troubleshooting Process**:
   - Run the code to identify missing packages
   - Install specific versions when encountering conflicts
   - For complex dependencies like DGL, be prepared to downgrade to versions compatible with the framework

5. **NumPy Consideration**:
   - NumPy version conflicts are common in ML environments
   - Consider using a middle-ground version that satisfies most dependencies

By following this systematic approach, we successfully created a working environment that can run the Mamba model within the LibCity framework despite the original environment issues. 