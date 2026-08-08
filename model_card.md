# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a random forest classification model with 100 decision
trees and a random state of 42. The model predicts whether an individual's
annual income is greater than $50,000 or less than or equal to $50,000.
Categorical features are transformed using one-hot encoding before training.
## Intended Use
The model was developed for an educational machine learning project. It is
intended to demonstrate data preprocessing, classification, model evaluation,
performance monitoring across data slices, and API deployment. It should not
be used to make employment, lending, insurance, or other high-impact decisions
about individuals.
## Training Data
The model was trained using the publicly available Census Income dataset. The
dataset contains demographic and employment-related features such as age,
education, occupation, work class, marital status, and hours worked per week.
The salary column is the target variable. Eighty percent of the dataset was
used for training. The categorical variables were one-hot encoded, and the
salary labels were converted to binary values.
## Evaluation Data
Twenty percent of the Census Income dataset was reserved for testing. A
stratified train-test split with a random state of 42 was used to preserve the
distribution of the salary classes. The test data was processed using the
encoder and label binarizer fitted only on the training data.
## Metrics
The model was evaluated using precision, recall, and F1 score. Precision
measures how often a prediction of income greater than $50,000 was correct.
Recall measures how many individuals earning greater than $50,000 were
identified. F1 provides a balance between precision and recall.

The model achieved the following results on the test data:

- Precision: 0.7353
- Recall: 0.6378
- F1 score: 0.6831

Performance was also measured for each unique value of every categorical
feature. These slice metrics are recorded in `slice_output.txt`.
## Ethical Considerations
The dataset contains sensitive demographic attributes, including age, race,
sex, marital status, and nationality. Historical patterns and inequalities in
the Census data may be learned and repeated by the model. Performance may also
differ across demographic groups, especially groups with fewer examples in the
dataset. A prediction from this model should not be treated as a reliable
assessment of an individual's ability, worth, or future income.
## Caveats and Recommendations

The dataset is historical and may not represent current populations, income
levels, occupations, or economic conditions. The model only identifies
patterns found in the available features and does not establish causal
relationships. Before any real-world use, the data should be updated, slice
performance should be reviewed for unequal outcomes, and fairness and bias
testing should be conducted. Additional model tuning and comparison with other
classification algorithms may also improve performance.