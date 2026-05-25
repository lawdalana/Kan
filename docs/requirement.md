This repo is an example code of Kan (kan: kolmogorov–arnold networks).
Using for simple classification task on python code.

Classification Task: House pricing predicting
Dataset:
- dataset/house_prices.arff
- dataset/openml_42165_metadata.json


In codes/original
- codes/original/model
  - contain the normal code to train and test MLP/DNN of Classification task
- codes/original/classify
  - contain the code that using model to classify task as API

In codes/kan
- codes/original/model
  - contain the normal code to train and test kan of Classification task
- codes/original/classify
  - contain the code that using model to classify task as API


In codes/kan-decode
- codes/original/model
  - contain the code that convert kan model to normal equation
- codes/original/classify
  - contain the code that using model to classify task as API


In codes/web-showcase
- contain static website that sample data and predict result comapre all model (DNN, Kan, Kan-equation).
- The result that show should contain latency as well.


Compare the result of this dataset on docs/result folder