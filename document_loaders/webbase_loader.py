from langchain_community.document_loaders import WebBaseLoader
url="https://www.amazon.in/Philips-NA342-7-2Litre-Through-window/dp/B0D9W76ZDV?ref_=ast_sto_dp&th=1&psc=1"
loader=WebBaseLoader(url)
docs=loader.load()
print(docs[0].page_content)