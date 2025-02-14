# Salary Classifier Project

## Overview

This project aims to classify individuals based on their income levels using a dataset containing various demographic and employment-related features. The dataset includes attributes such as age, workclass, education, occupation, and more. The goal is to build a machine learning model that can predict whether an individual's income is above or below a certain threshold.

## Dataset

The dataset used in this project is named `adult.csv` and contains the following columns:

- `age`: Age of the individual.
- `workclass`: Type of employment.
- `fnlwgt`: Final weight.
- `education`: Highest level of education achieved.
- `education_num`: Numeric representation of education level.
- `martial_status`: Marital status.
- `occupation`: Type of occupation.
- `relationship`: Relationship status.
- `race`: Race of the individual.
- `gender`: Gender of the individual.
- `capital_gain`: Capital gains.
- `capital_loss`: Capital losses.
- `hours_per_week`: Hours worked per week.
- `country`: Country of origin.
- `income`: Income level (<=50K or >50K).

## Project Structure

The project is structured as follows:

1. **Importing Libraries**: Essential libraries such as pandas, numpy, matplotlib, seaborn, and various scikit-learn modules are imported.
2. **Loading Data**: The dataset is loaded into a pandas DataFrame.
3. **Data Exploration**: Basic exploration of the dataset, including value counts for each column and data preview.
4. **Data Preprocessing**: Handling duplicates, missing values, and encoding categorical variables.
5. **Data Visualization**: Visualizing the distribution of various features and their relationship with the target variable (`income`).
6. **Model Building**: Implementing machine learning models such as Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Decision Tree, Logistic Regression, and Random Forest.
7. **Model Evaluation**: Evaluating model performance using metrics such as accuracy, confusion matrix, and classification report.
8. **Saving Models**: Saving trained models using joblib and pickle for future use.

## Requirements

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly
- joblib
- pickle

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly joblib pickle
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/salary-classifier-project.git
   cd salary-classifier-project
   ```

2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Copy_of_Salary_classifier_project.ipynb
   ```

3. **Follow the Notebook**: Execute the cells in the notebook to load the data, preprocess it, build models, and evaluate their performance.

## Results

The project includes various visualizations and model evaluations to understand the dataset and the performance of different classifiers. Key visualizations include:

- Distribution of income levels.
- Relationship between race and income.
- Gender distribution and its impact on income.

Model performance is evaluated using accuracy scores, confusion matrices, and classification reports.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project is sourced from the UCI Machine Learning Repository.
- Special thanks to the open-source community for providing valuable resources and libraries.

## Contact

For any questions or feedback, please contact Roaa at roaa.hazem.isamil@gmail.com.

---

This README provides a comprehensive overview of the project, its structure, requirements, and usage instructions. It also includes sections for results, contributing, license, acknowledgments, and contact information.
