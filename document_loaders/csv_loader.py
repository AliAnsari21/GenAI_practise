from langchain_community.document_loaders import CSVLoader
loader=CSVLoader(file_path=r"E:\end to end projects\multiple disease prediction\Data\Diabetes_prediction.csv")
data=loader.load()
print(data[0])