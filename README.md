# Condition-Number Adaptive-Weight PINN (A-PINN) for Stewart Platforms

This repository contains the official implementation for the paper **"Condition-Number Adaptive-Weight PINN (A-PINN): A High-Fidelity and Real-Time Forward Kinematics Solver for Stewart Platforms"**. 

Parallel kinematic mechanisms (PKMs) require accurate, low-latency, and deterministic estimation of the end-effector pose for high-bandwidth closed-loop control. However, solving forward kinematics (FK) is challenging because the strongly coupled nonlinear closed loop renders the closed-form solutions non-unique, and iterative numerical solvers are prone to ill-conditioning near singular configurations. 

This project provides a $\kappa$-Adaptive Physics-Informed Neural Network (A-PINN) that outputs pose- and Jacobian-consistent velocity-mapping quantities in a single forward pass, enabling deterministic low-latency inference.

## 🌟 Key Features and Contributions

*   **Implicit Sobolev Regularization:** The framework proposes an operator-consistency loss to ensure high-fidelity Jacobian learning without the need for unstable matrix inversion. The training objective enforces an implicit Sobolev-type regularization via operator consistency.
*   **$\kappa$-Adaptive Weighting:** The physical loss is dynamically modulated by a weighting mechanism driven by the Local Conditioning Index (LCI) or $1/\kappa$. This condition-number-aware strategy robustly constrains ill-conditioned regions and prevents boundary underfitting.
*   **Real-Time Validation:** The proposed A-PINN is validated on a physical 6-UPS platform with a 1 kHz sampling rate. It achieves 1 kHz closed-loop control with an inference latency of 0.141 ms and a relative Jacobian error of 0.40%.

## 📂 Repository Structure

Based on the project files, the repository is organized as follows:

*   **`Data_Generation/`**: Contains scripts for generating high-fidelity datasets.
    *   `Data_Generator_Normalized_Ha...`: Script used to create a spatially uniform training dataset via Inverse Transform Sampling.
*   **`Train_model/`**: Contains the core training implementation.
    *   `L2ori_wJ_aPINN.py`: The main Python script for training the A-PINN framework.
    *   **`result/`**: Stores outputs from the training process.
        *   `L2ori_wJ_adaptive_PINN.pt`: The trained PyTorch model weights.
        *   `result_info.txt`: Text file logging the training results and information.
*   `.gitignore`: Specifies intentionally untracked files to ignore.
*   `README.md`: This documentation file.

## 🚀 Getting Started / Usage

### 1. Prerequisites
Ensure you have Python installed along with the necessary deep learning and scientific computing libraries (e.g., PyTorch, NumPy) to run the neural network models and dataset generators.

### 2. Data Generation
To train the A-PINN model, you first need to generate the high-fidelity training dataset, which utilizes an Inverse Transform Sampling strategy to correct density distribution biases. Navigate to the data generation directory and run the script:

```bash
cd Data_Generation
# Run the dataset generation script (replace with the full filename)
python Data_Generator_Normalized_Haar_Sobol_Rejection_Sampling.py 
```

### 3. Model Training
Once the dataset is prepared, you can train the A-PINN framework. The training script automatically applies the condition-number-adaptive weights and implicit Sobolev regularization to ensure physical consistency. 

⚠️ IMPORTANT: Before starting the training, please open `L2ori_wJ_aPINN.py` and modify the dataset path (at the end of the script) variable to match the actual location of the dataset you just generated on your local machine.

```bash
cd ../Train_model
python L2ori_wJ_aPINN.py
```
Upon successful training, the PyTorch model weights will be saved as `L2ori_wJ_adaptive_PINN.pt` inside the `Train_model/result/` directory. The training logs will be saved in `result_info.txt`.

## 📊 Demo Vedio

https://github.com/user-attachments/assets/339e219e-859d-4bbe-a369-ca24849c2064



## 📖 Citation

If you use this code or our methodology in your research, please consider citing the associated paper:

```bibtex
TBD
```
