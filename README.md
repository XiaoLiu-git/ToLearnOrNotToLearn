# ToLearnOrNotToLearn
Here is code for our NeurIPS2024 paper ***To Learn or Not to Learn, That is the Question — A Feature-Task Dual Learning Model of Perceptual Learning***.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Now this repository only contains the reproduction of the double-training experiment (without visualization).
Code for other results in the paper will be updated soon.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/XiaoLiu-git/ToLearnOrNotToLearn.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## File Descriptions

| File | Description |
| ---- | ----------- |
| **run_double_training.py** | Main script that reproduce the double-training experiment results (without visualization). |
| **image.py** | Contains helper functions for image stimulus generation. |
| **network.py** | Defines functions related to the neural network architecture. 
| **tool_gabor.py** | Provides general Gabor filter tools. |
| **utils.py** | Contains miscellaneous utility functions used across the project, such as file handling, logging, and other common tasks that simplify coding and improve code readability. |

## Reproduce Figures in the Paper

1. First Experiment

    (If you have already run the fourth experiment data, you can reuse it.)
    Data preparing:
    ```bash
    python run_double_training.py --save_path=./result/first/
    ```

2. Second Experiment

    Data preparing:
    ```bash
    python .py
    ```

3. Third Experiment

    Data preparing:
    ```bash
    python run_double_training.py --noise_cutout=2 --conventional_epoch=40 --save_path=./result/third/2sessions/
    python run_double_training.py --noise_cutout=2 --conventional_epoch=80 --save_path=./result/third/4sessions/
    python run_double_training.py --noise_cutout=2 --conventional_epoch=160 --save_path=./result/third/8sessions/
    python run_double_training.py --noise_cutout=2 --conventional_epoch=240 --save_path=./result/third/12sessions/
    ```

4. Fourth Experiment

    Data preparing:

    (If you have already run the first experiment data, you can reuse it.)
    ```bash
    python run_double_training.py --save_path=./result/fourth/
    ```

## Usage

1. Run the main script:
    ```bashs
    python db_after_threshold_pretrain_2cnn_sepa.py
    ```
2. (Optional) Customize parameters in each module as needed.

    To be updated...

## Examples

To be updated...


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


