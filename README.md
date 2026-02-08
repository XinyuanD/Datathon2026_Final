# Predictive Analysis of U.S. Health Access: ML Prediction with XGBoost
(Tools used: NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib)

## 1. Explanation of Process
We trained a machine learning model that treats U.S. health disparities as a time-series forecasting problem. The core of our process involved transforming raw health estimates into Z-scores normalized by topic, allowing us to compare disparities across different medical taxonomies.

We implemented a Temporal Walk-Forward split: training the model strictly on 2019–2023 data to predict the 2024 outcomes. This approach prevents data leakage and ensures the model is learning long-term trends rather than memorizing specific years. We iterated through multiple architectures, including CatBoost and Gradient Boosting, before finally deciding to use XGBoost. During the process, data leakage issues occurred, where either 2024 data was being mixed into the training data, or the model was memorising the data as it progressed, so precision model engineering was required.

## 2. Quality of Data Exploration
Our data exploration focuses on understanding the structure, scale, and behavior of the dataset before modelling. We first examined how the data was organized across ‘TOPIC’, ‘SUBGROUP’, and ‘TIME_PERIOD’, showing that it formed a hierarchical time series. This led us to use a temporal train-test split.

We found large variations in magnitude in scale across health topics, which led us to normalize the target variable using the Z-scores within the topic. This helped the model to learn relative changes and trends rather than being biased by absolute value differences between topics. We also examined ‘SUBGROUP’ labels to identify key socioeconomic factors, including disability status, poverty, insurance coverage, and minorities, which preserve important context for the data. 

Lastly, we evaluated different features after training to see if they contributed meaningfully to the predictions.

## 3. Model specifications
We implemented an Optimized XGBoost Regressor with built-in categorical support.

- Architecture: The model utilizes a Gradient Boosted Decision Tree (GBDT) structure with a histogram-based tree-growing method for high-speed processing of the large NHIS feature space.
- Optimization: To prevent overfitting, we implemented Early Stopping (at the 449th iteration) and a reduced learning rate (0.01). This forced the model to prioritize generalizable "social trends" over specific data noise, as well as made sure the model did not memorize data from other 2024 data and bring it into later predictions.

## 4. Quality of Model (Features & Error)
Our model achieved a robust balance between complexity and predictive power, using engineered features such as z-score, z-delta, and rolling_3yr_z.

Performance Metrics:
- **R2-Score: 0.82**
  -An R2 score of 0.82 indicates that our model explains 82% of the variance in health disparities. It is high enough to be a reliable forecasting tool, but lower than an 'identity-leakage' model that achieves an R2 score of 0.98. This means that our model has learned generalizable trends rather than just disease-specific constants.
- **RMSE: 0.43**
  -With a standard deviation of 1.0 for our Z-score target, an RMSE of 0.43 confirms that our predictions typically fall within less than half a standard deviation of the actual 2024 reported disparities.

## 5. Value of the Model
- Trend Forecasting: By utilizing the z_delta (momentum) feature, the model identifies not only where health gaps exist but also where they are widening fastest, enabling targeted policy interventions.
- Intervention Simulation: The model quantifies the predictive power of social determinants, showing that a group’s historical trend is a stronger predictor of future access than the specific disease type itself.
- Scalability: The framework is designed to ingest new CDC data annually, automatically updating the "Moving Average" and "Momentum" features to provide continuous monitoring of U.S. healthcare equity.
