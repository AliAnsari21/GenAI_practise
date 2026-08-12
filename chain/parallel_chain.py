from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence
)
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

template1=PromptTemplate(
    template='Generate short and simple notes from following text {text}',
    input_variables=['text']
)

template2=PromptTemplate(
    template='generate 5 question answer from the following text.\n {text}',
    input_variables=['text']
)

template3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()
parallel_chain=RunnableParallel({
    'notes':template1|model|parser,
    'quiz':template2|model|parser
})

merge_chain=template3|model|parser

chain=parallel_chain|merge_chain
text="""
## Machine Learning

**Machine Learning (ML)** is a branch of Artificial Intelligence (AI) that enables computers to learn patterns and relationships from data and make predictions or decisions without being explicitly programmed for every individual task. Instead of writing fixed rules for every possible situation, we provide a machine learning algorithm with historical data, and the algorithm learns from that data to produce a model. This model can then be used to make predictions on new, unseen data. For example, in a house price prediction system, we can provide information such as location, area, number of bedrooms, age of the house, and previous selling prices. The ML model learns the relationship between these features and the price and can then predict the price of a new house.

The basic idea behind machine learning is **learning from data**. Data generally contains **features** and, in supervised learning, a **target variable**. Features are the input variables used by the model, while the target is the value that we want the model to predict. For example, in a student performance dataset, study hours, attendance, and previous marks can be features, while the final examination score can be the target. The machine learning algorithm analyzes the available examples and identifies patterns that can be generalized to new examples.

Machine learning is generally divided into **three major types: Supervised Learning, Unsupervised Learning, and Reinforcement Learning**. In **Supervised Learning**, the model is trained using labeled data, meaning that both input features and the correct output are available. The model learns the relationship between the inputs and outputs and uses this relationship to predict outputs for new data. Supervised learning is mainly divided into **classification and regression**. Classification is used when the output belongs to a category, such as predicting whether an email is spam or not spam, whether a customer will churn or not, or whether a patient belongs to a particular disease category. Regression is used when the output is a continuous numerical value, such as house price, salary, temperature, or sales amount.

Common supervised learning algorithms include **Linear Regression, Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes, Gradient Boosting, AdaBoost, XGBoost, and CatBoost**. Linear Regression is commonly used for predicting continuous numerical values. Logistic Regression is commonly used for binary classification problems. Decision Trees make predictions using a tree-like structure of conditions. Random Forest combines multiple decision trees to improve prediction performance and reduce overfitting. Support Vector Machines try to find an optimal boundary between classes, while KNN predicts the output based on nearby training examples. Boosting algorithms such as XGBoost and CatBoost build models sequentially to improve performance and are widely used for structured or tabular datasets.

In **Unsupervised Learning**, the dataset does not contain a predefined target variable. The algorithm attempts to discover hidden patterns, structures, or groups within the data. One of the most common unsupervised learning techniques is **clustering**, where similar data points are grouped together. For example, a company may use customer information such as age, spending habits, and purchase frequency to divide customers into different groups. Popular clustering algorithms include **K-Means Clustering, Hierarchical Clustering, and DBSCAN**. Another important area is **dimensionality reduction**, where the number of features is reduced while attempting to preserve important information. **Principal Component Analysis (PCA)** is one of the most commonly used dimensionality-reduction techniques.

**Reinforcement Learning (RL)** is another type of machine learning in which an agent learns by interacting with an environment. The agent performs actions and receives rewards or penalties based on those actions. The objective is to learn a strategy, called a **policy**, that maximizes the total reward over time. Reinforcement learning is used in areas such as robotics, game playing, autonomous systems, recommendation systems, and resource management. Algorithms such as **Q-Learning, SARSA, and Deep Q-Networks (DQN)** are examples of reinforcement learning approaches.

A typical machine learning project begins with **problem definition**. Before selecting an algorithm, it is important to understand what problem needs to be solved and what output is expected. The next step is **data collection**, where relevant data is obtained from sources such as databases, CSV files, APIs, websites, sensors, or public datasets. After collecting the data, the next important stage is **data preprocessing**. Real-world datasets often contain missing values, duplicate records, incorrect data types, inconsistent values, and outliers. These problems need to be addressed before training a model.

**Exploratory Data Analysis (EDA)** is performed to understand the dataset and identify relationships, trends, distributions, and unusual observations. During EDA, techniques such as descriptive statistics, correlation analysis, histograms, box plots, scatter plots, and bar charts can be used. Libraries such as **Pandas, NumPy, Matplotlib, and Seaborn** are commonly used for data analysis and visualization in Python. EDA helps determine which variables may be useful for prediction and can also reveal potential data-quality problems.

After preprocessing and EDA, the next step is **feature engineering and feature selection**. Feature engineering involves creating useful new variables from existing data. For example, from a date column, we can create year, month, day, or weekday features. Feature selection involves choosing the most relevant features for the model and removing unnecessary or highly redundant variables. Good feature engineering can significantly improve model performance.

The dataset is commonly divided into **training and testing datasets**. The training dataset is used to teach the model, while the testing dataset is used to evaluate how well the trained model performs on unseen data. A common split is 80% training data and 20% testing data, although the appropriate split depends on the problem and dataset. For some projects, **validation data or cross-validation** is also used to select models and tune hyperparameters without using the final test set for decision-making.

Some machine learning algorithms require **feature scaling** because they are sensitive to the magnitude of input variables. For example, if one feature ranges from 0 to 1 and another ranges from 0 to 100,000, the larger feature can disproportionately influence certain algorithms. Techniques such as **Standardization** and **Min-Max Normalization** are commonly used. Standardization transforms features so they generally have a mean of approximately zero and a standard deviation of approximately one, while Min-Max scaling typically transforms values into a specified range such as 0 to 1.

After preparing the data, we select an appropriate **machine learning algorithm** and train the model using the training data. During training, the algorithm attempts to learn parameters that minimize errors or maximize an appropriate objective. Once training is complete, the model can generate predictions for unseen data. The model must then be evaluated using appropriate **evaluation metrics**.

For regression problems, commonly used evaluation metrics include **Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score**. MAE measures the average absolute difference between actual and predicted values. MSE calculates the average squared error and gives greater importance to larger errors. RMSE is the square root of MSE and is expressed in the same units as the target variable. R² indicates how much of the variation in the target variable is explained by the model.

For classification problems, common metrics include **Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix**. Accuracy represents the proportion of correct predictions, while precision measures how many predicted positive cases are actually positive. Recall measures how many of the actual positive cases were correctly identified. F1-score combines precision and recall into a single metric. A confusion matrix provides a detailed breakdown of correct and incorrect classification results.

One of the most important challenges in machine learning is **overfitting**. Overfitting occurs when a model learns the training data too closely, including noise and random patterns, resulting in excellent training performance but poor performance on unseen data. The opposite problem is **underfitting**, where the model is too simple to capture the important patterns in the data. Techniques such as cross-validation, regularization, feature selection, pruning, early stopping, and appropriate model complexity can help address these problems.

**Hyperparameter tuning** is another important part of machine learning. Hyperparameters are settings chosen before or during model training rather than learned directly from the training data. Examples include the number of trees in a Random Forest, learning rate in boosting algorithms, maximum tree depth, and the value of K in KNN. Techniques such as **Grid Search, Random Search, and Bayesian optimization** can be used to find suitable hyperparameter values.

After selecting and evaluating the final model, it can be **saved and deployed** so that other users or applications can use it. A trained Python model can commonly be saved using tools such as **Joblib or Pickle**, depending on the model and project requirements. The model can then be integrated into an application using frameworks such as **Flask, FastAPI, or Streamlit**. For example, a house price prediction model can be deployed through a web application where users enter location, area, BHK, and other information and receive a predicted price.

A complete machine learning workflow can therefore be summarized as **Problem Definition → Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Feature Selection → Train/Test Split → Model Selection → Model Training → Model Evaluation → Hyperparameter Tuning → Final Model → Deployment → Monitoring**. This workflow is not always strictly linear; in practical projects, data scientists often move back and forth between different stages when they discover new problems or insights.

Machine learning has applications across many industries. In **finance**, it can be used for fraud detection, credit scoring, and risk prediction. In **healthcare**, it can assist with disease-risk prediction and medical-image analysis. In **retail**, it can be used for recommendation systems, customer segmentation, and demand forecasting. In **manufacturing**, ML can support predictive maintenance and quality control. In **transportation**, it can be used for route optimization and autonomous systems. In **natural language processing**, machine learning supports applications such as sentiment analysis, text classification, translation, and conversational AI.

For someone learning machine learning, **Python** is one of the most commonly used programming languages because of its extensive ecosystem. **NumPy** is useful for numerical computation, **Pandas** for data manipulation, **Matplotlib and Seaborn** for visualization, and **Scikit-learn** for traditional machine learning algorithms and preprocessing. For advanced machine learning and deep learning, frameworks such as **TensorFlow and PyTorch** are widely used. Tools such as **Jupyter Notebook, VS Code, Git, and GitHub** are also commonly used during development and project management.

In simple terms, **Machine Learning means teaching a computer to learn useful patterns from data so that it can make predictions or decisions on new data**. A strong ML project is not only about choosing an algorithm; it involves understanding the problem, collecting quality data, cleaning and analyzing it, selecting useful features, training appropriate models, evaluating them correctly, preventing overfitting, and finally deploying the model so it can provide value in a real-world application.

"""
result=chain.invoke({'text':text})
print(result)