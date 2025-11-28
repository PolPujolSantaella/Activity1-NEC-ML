# Activity 1: Prediction with Supervised Learning Models

## 🎯 Project Overview

This repository contains the implementation, analysis, and documentation for Activity 1 of the **Neural and Evolutionary Computation (NEC)** course (2025/26). The primary goal of this assignment is to apply and compare different supervised learning models—including a custom-built Neural Network—to a regression task using the "Realistic Revenue Dataset."

The analysis focuses on exploring data preprocessing techniques, hyperparameter tuning, and comparing performance across various modeling approaches.

## 📚 Models and Techniques Implemented

The activity covers both mandatory and optional sections, demonstrating proficiency in several core Machine Learning and Deep Learning concepts:

### Mandatory Implementation
* **Multiple Linear Regression (MLR-F):** Implemented using the `scikit-learn` library.
* **Backpropagation Neural Network (BP):** Implemented **from scratch** based on the mathematical formulation of the course materials (for a single hidden layer).
* **Library Backpropagation (BP-F):** Implemented using the **PyTorch** deep learning framework for comparison.

### Optional Parts (Completed)
* **Optional Part 1: Study the effect of the different regularization techniques in the Neural Network used in the (BP-F)**: Implementation of Neural Network model with PyTorch using L2 Regulariztion and Dropout.

* **Optional Part 3: Ensemble Learning Models:** Implementation and evaluation of at least two ensemble techniques (e.g., Random Forest, Gradient Boosting) to further improve predictive performance.

## 💾 Repository Structure

| Directory/File | Description |
| :--- | :--- |
| `data/` | Contains the input dataset (`Realistic_Revenue_Dataset.csv`) used for training and testing. |
| `src/` | Python Implementation of the Neural Network with Back Propagation implemented by me.|
| `notebooks/` | Jupyter Notebooks detailing the step-by-step analysis, preprocessing, hyperparameter tuning, and results comparison. This is the primary location for the executable code and figures. |
| `docs/` | Supporting documentation, including the final report and presentation slides (if any). |
| `NEC_Activity1_PujolSantaella_Pol.pdf` | **Official Activity Report** detailing all methodology, analysis, and conclusions. |

## ⚙️ How to Run the Code

To replicate the experiments and results presented in the report, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PolPujolSantaella/Activity1-NEC-ML.git](https://github.com/PolPujolSantaella/Activity1-NEC-ML.git)
    cd Activity1-NEC-ML
    ```

2.  **Set up the environment:**
    The project relies on standard scientific Python libraries. A `requirements.txt` file (or similar) is recommended for dependencies.
    * **Key Libraries:** Python 3.x, NumPy, Pandas, Scikit-learn, PyTorch (or TensorFlow), Matplotlib.
    * It is highly recommended to use a virtual environment.

3.  **Run the notebooks:**
    Execute the main analysis notebook (usually located in the `notebooks/` directory) in sequence to run the data loading, preprocessing, model training, and evaluation steps.

## 📄 Final Report

For a detailed analysis of the methodology, hyperparameter selection (including the best $16:32:1$ configuration with $\text{Sigmoid}$ activation), results comparison, and discussion, please refer to the final report:

* **[NEC\_Activity1\_PujolSantaella\_Pol.pdf](./docs/NEC_Activity1_PujolSantaella_Pol.pdf)

## 🧑‍💻 Author

| Name | Student ID |
| :--- | :--- |
| Pol Pujol Santaella |

**Course:** Neural and Evolutionary Computation (NEC) - 2025/26