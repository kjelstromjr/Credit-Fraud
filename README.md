# Machine Learning Algorithms for Credit Card Fraud

This is a project that tests 4 machine learning models on a dataset with credit card transactions that are marked as normal or fraud. The tested models are logisitic regression, decision tree, a tuned decision tree, and SVM. The purpose of this project is to determine which algorithm performs better and which features have the most impact on the outcome.

This repository contains the code to train and test the models as well as a powerpoint and pdf report outlining my results.

### Running the code

1. Install all packages
    ```
    pip install -r requirements.txt
    ```

2. Run the code
    > This will download the dataset, train the models, and display the results
    ```
    python model.py
    ```

### Viewing the results

All of the number results will be printed in the console with each corresponding test. The program also creates two images `precision_recall_curves.png` and `roc_curves.png`. If you want to see an example of the images, take a look in the "Evalutation and Results" secion of my report.