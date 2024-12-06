# ToLearnOrNotToLearn
Here is code for our NeurIPS2024 paper ***To Learn or Not to Learn, That is the Question — A Feature-Task Dual Learning Model of Perceptual Learning***.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
| **run_double_training.py** | Main script that reproduce the exp 1, 3, 4 results (without visualization). The usage is in the next section.|
| **run_random_rotating.py** | Main script that reproduce the exp 2 results (without visualization). The usage is in the next section.|
| **image.py** | Contains helper functions for image stimulus generation. |
| **network.py** | Defines functions related to the neural network architecture. 
| **tool_gabor.py** | Provides general Gabor filter tools. |
| **utils.py** | Contains miscellaneous utility functions used across the project, such as file handling, logging, and other common tasks that simplify coding and improve code readability. |
| **visualize_first_fourth_experiment.ipynb** | Visualize exp 1 & 4 after running **run_double_training.py**. The usage is in the next section.|
| **visualize_second_experiment.ipynb** | Visualize exp 2 after running **run_random_rotating.py**. The usage is in the next section.|
| **visualize_third_experiment.ipynb** | Visualize exp 3 after running **run_double_training.py**. The usage is in the next section.|

## Reproduce Figures in the Paper

1. First Experiment
    
    Data preparing:
    
    (If you have already run the fourth experiment data, you can reuse it. Here we only see the conventional training part.)
    ```bash
    python run_double_training.py --save_path=./result/first/
    ```
    Visualization:

    Run all the code in ```visualize_first_fourth_experiment.ipynb``` to get exp1_threshold_100.svg and exp1_improvement_100.svg. (You can also get the figures for the fourth experiment.)

2. Second Experiment

    Data preparing:
    ```bash
    <!-- for random -->
    python run_random_rotating.py --training_mode=random --save_path=./result/second/random/
    <!-- for rotating -->
    python run_random_rotating.py --training_mode=rotating --save_path=./result/second/rotating/
    ```
    Run all the code in ```visualize_second_experiment.ipynb``` to get exp2_random_threshold_100.svg, exp2_rotating_threshold_100.svg and exp2_improvement_100.svg. 

3. Third Experiment

    Data preparing:
    ```bash
    python run_double_training.py --noise_cutout=2 --l_lambda_pre=3 --l_lambda=3 --conventional_epoch=40 --save_path=./result/third/2sessions/
    python run_double_training.py --noise_cutout=2 --l_lambda_pre=3 --l_lambda=3 --conventional_epoch=80 --save_path=./result/third/4sessions/
    python run_double_training.py --noise_cutout=2 --l_lambda_pre=3 --l_lambda=3 --conventional_epoch=160 --save_path=./result/third/8sessions/
    python run_double_training.py --noise_cutout=2 --l_lambda_pre=3 --l_lambda=3 --conventional_epoch=240 --save_path=./result/third/12sessions/
    ```
    Visualization:

    Run all the code in ```visualize_third_experiment.ipynb``` to get the student t-test result, exp3_threshold_100.svg and exp3_improvement_100.svg. 
4. Fourth Experiment

    Data preparing:

    (If you have already run the first experiment data, you can reuse it. It also has the double training part data, just only analyse the conventional training part.)
    ```bash
    python run_double_training.py --save_path=./result/fourth/
    ```
    Visualization:

    Run all the code in ```visualize_first_fourth_experiment.ipynb``` to get exp4_threshold_100.svg and exp4_improvement_100.svg. You will also get get exp1_threshold_100.svg and exp1_improvement_100.svg.You need to combine the exp1_improvement_100.svg and exp4_improvement_100.svg to get the figure in the paper.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


