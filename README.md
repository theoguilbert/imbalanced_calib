# Calibration Methods in Imbalanced Binary Classification

Data and code for reproducibility of experiments of the "Calibration Methods in Imbalanced Binary Classification" paper.

To run experiments, please first install the given requirements via the pip install -r requirements.txt command. Then you can run the **calibration_analysis** Jupyther Notebook.

Here is the description of the folders and files:
* Data used for experiments are in the **datas** folder.
* The results of experiments presented in the paper are available in the **results_RF** folder.
* **calibration_analysis**: Main file (Jupyter Notebook) that has been used to generate the experiments. It can be used to generate new results for fair reproducibility.
* **calibration_evaluation**: In this file, you can find functions to generate, extract and retrieve calibration results.
* **create_datasets**: It is the file where we define functions to generate binary imbalanced datasets that are used in our experiments.
* **calibrated_model**: We define the classification model as well as the different calibration methods.
* **statistical_test**: In this file, we define statistical tests and the process to assess significant differences in the different calibration methods used.
* **tools_and_calib_metrics**: We define some tools and the different calibration metrics in this file.
