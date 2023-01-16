# thermal-dosimetry-surrogate

The repository contains the code for figures/results in work-in-progress papers for the URSI GASS 2023 and BioEM 2023 conference.


To reproduce the results, easiest way is to create a local environment by using `conda` as
```shell
conda create --name thermal-dosimetry-surrogate python=3.9.12
```
and, inside the environment, within `code` repository, run the following command
```shell
pip install -r requirements.txt
```
to install all dependencies listed in `requirements.txt`.

## Contents

| Directory | Subdirectory/Contents | Description |
|:---:|:---:|:---:|
| `data` |  |  |
| 1 | raw | Collected from the annex of the IEEE Std 2889-2021. |
| 2 | processed | Clean version of the collected data and the synthetic data set. |
| 3 | models | Results regarding the predictive performance of surrogate models. |
| `figures` |  | Generated figures and further augmented figures for conference papers. |
| `models` |  | Parameters of fitted surrogate models. |
| `notebooks` |  | Jupyter notebooks. |
| 1 | 00_data_processing.ipynb | Cleaning raw data, initial visualizations. |
| 2 | 01_synthetic_data_generation.ipynb | Generating the synthetic data set. |
| 3 | 02_baseline_model.ipynb | XGBoost baseline surrogate model. |
| 4 | 03_multilayer_perceptron.ipynb | Feedforward neural network surrogate model. |
| 5 | 04_mixture_of_experts.ipynb | Quadratic polynomial + tensor product splines. |
| 6 | 05_tabnet.ipynb | TabNet surrogate model. |
| 7 | 06_postprocessing.ipynb | Visualization of the predictive performance of surrogate models. |
| `src` |  | Code used in 03_multilayer_perceptron.ipynb. |

 ## License

 [MIT](https://github.com/antelk/thermal-dosimetry-surrogate/blob/main/LICENSE)

 ## Author

Ante Kapetanović
