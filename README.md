# Product Classification

This project is about classifying products using Machine Learning. The main script is `main.py`.

## Requirements

- Python 3.6 or higher
- Libraries: sklearn, pandas, numpy, argparse, time, pickle, os

## Data
My validation data is provided by my lecturer, it located at `data/validate.txt`. And I get the label from this then I use it for crawling data for training. 

I scrape from `tiki` and `vatgia` website. You can try to crawl the data following my repo: [Crawl Data]
(https://github.com) 


The data used for training in this project is located in `data/train.txt`. Each line in the file represents a 
product with its category and description. 

For testing, the data is located in `data/test.txt`. Each line in the file represents a product with its category and description.

The data is in the following format:

```
__label__cate_gory description
```

## Usage

To run the script, use the following command:

```shell
bash train.sh
```

## Project Structure

The project consists of several Python scripts:  

- train.sh: This shell script sets up the environment and runs the train.py script with the necessary arguments.
- train.py: This script is responsible for loading the data, preprocessing it, splitting it into training and testing sets, training a Naive Bayes model, and evaluating the model's performance.

The project uses a Naive Bayes model for product classification. The model is trained on the product data and then evaluated for its performance. The trained model and the vectorizer are saved as pickle files for future use.

## Workflow
The workflow of the project is as follows:  
1. Load the training data from data/train.txt.
2. Preprocess the training data (if the --is_preprocess flag is set to True).
3. Encode the target labels into numerical form.
4. Train a Naive Bayes model on the training data (if the --is_train flag is set to True).
5. Evaluate the model's performance on the validate data (if the --is_evaluate flag is set to True).
6. Save the trained model and the vectorizer as pickle files.

The project is designed to be flexible, allowing you to control the preprocessing, training, and evaluation stages through command-line arguments.

## License
[MIT](https://choosealicense.com/licenses/mit/)
```