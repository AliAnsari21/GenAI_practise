from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text_splitter=SemanticChunker(
    embedding,breakpoint_threshold_type='standard_deviation',breakpoint_threshold_amount=1
)

sample="""
Artificial Intelligence (AI) is one of the most important and rapidly developing areas of computer science. It focuses on creating machines and software systems that can perform tasks that normally require human intelligence. These tasks include understanding natural language, recognizing images, making decisions, solving problems, learning from data, and predicting future outcomes. Artificial Intelligence has become an important part of modern technology and is being used in healthcare, finance, education, transportation, entertainment, manufacturing, agriculture, and many other industries.

Machine Learning is a major branch of Artificial Intelligence. Instead of explicitly programming a computer with rules for every possible situation, Machine Learning allows computers to learn patterns from data. A machine learning model is trained using historical data and then uses the learned patterns to make predictions or decisions on new data. There are several types of machine learning, including supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning is used when the training dataset contains both input features and the correct output labels. The model learns the relationship between the input and output and attempts to predict the correct output for new examples. Common supervised learning algorithms include Linear Regression, Logistic Regression, Decision Trees, Random Forest, Support Vector Machines, K-Nearest Neighbors, Gradient Boosting, and Neural Networks. Supervised learning can be used for both classification and regression problems.

Classification is a machine learning task where the goal is to predict a category or class. For example, an email classification system can determine whether a message is spam or not spam. A medical model can classify a patient into different disease categories based on medical information. A financial system can classify transactions as legitimate or fraudulent. Classification models are evaluated using metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.

Regression is another supervised learning task where the goal is to predict a continuous numerical value. For example, a housing price prediction system can estimate the price of a house based on its location, size, number of bedrooms, age, and other characteristics. Other examples include predicting sales, temperature, stock prices, energy consumption, and customer spending. Regression models can be evaluated using metrics such as Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and R-squared.

Unsupervised learning is used when the dataset does not contain predefined output labels. The algorithm attempts to discover hidden structures and patterns within the data. Clustering is one of the most common unsupervised learning techniques. K-Means clustering, Hierarchical clustering, and DBSCAN can be used to group similar data points together.
"""
docs=text_splitter.create_documents([sample])
print(docs)