from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(r'C:\Users\Ali Ansari\OneDrive\Desktop\Ansari_Ali_DataScience_Resume 1.pdf')
docs=loader.load()
print(docs)