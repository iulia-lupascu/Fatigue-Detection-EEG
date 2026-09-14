## Driving fatigue detection through transfer learning

### Description
This project involves an EEG-based method for detecting driver drowsiness using a hybrid LSTM-Transformer architecture. 

### Dataset
The dataset is comprised of .mat files representing by EEG recordings of 3-second increments, collected from 11 subjects at 128 Hz across 30 channels, resulting in  2022 samples. Each sample has a corresponding unique index and binary fatigue indicator, therefore the dataset is labeled.

### Preprocessing
The data has been processed by a 1-Hz high-pass and 50-Hz low-pass FIR filter and automatic artefact rejection. Z-score normalization was implemented across each subject over all of the trials to achieve a consistent sample distribution across subjects.

### Methodology 
Supervised learning was implemented to classify whether a subject is drowsy or alert by developing a hybrid LSTM-Transformer architecture. The EEGTransformerModel class defines an architecture that begins with a linear transformation to project EEG channel data into the model's internal feature space, followed by an LSTM network for capturing the short-term temporal variations in the EEG signals. A Transformer Encoder is utilized to acquire the global dependencies and long-range patterns. During training, noise was added to improve generalization. Next, Leave-one-out cross-validation (LASO-CV) was carried out to discover the parameters that perform the best on the dataset.

### Evaluation
Given the nature of the task, which requires predicting a binary outcome, the following evaluation metrics were employed: accuracy, precision, recall, and F1 score for obtaining the model performance, and ROC-AUC for quantifying the true and false positive rates.

### Results
After performing hyperparameter tuning, the model achieved an accuracy of 76.8%, a strong precision score of 83.3%, indicating reliable idenfication of fatigue with minor false alarms, a recall of 69.4%, and an F1 score of 0.74.
