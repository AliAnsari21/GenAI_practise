from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Create embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    Document(
        page_content="""
Virat Kohli is one of India's most prominent cricketers and has been a major figure in the Indian Premier League. He has represented Royal Challengers Bengaluru (RCB) throughout his IPL career. Kohli is primarily a right-handed top-order batsman and is known for his consistency, aggressive batting, fitness, and ability to build long innings. He has also served as captain of RCB for several seasons.
""",
        metadata={
            "player": "Virat Kohli",
            "team": "RCB",
            "role": "Batter"
        }
    ),

    Document(
        page_content="""
Rohit Sharma is an Indian international cricketer and one of the most successful captains in IPL history. He has represented Mumbai Indians (MI) and played a major role in the team's success. Rohit is a right-handed opening batter known for his timing, six-hitting ability, and ability to score large innings. He has also been an important leadership figure for Mumbai Indians.
""",
        metadata={
            "player": "Rohit Sharma",
            "team": "MI",
            "role": "Batter"
        }
    ),

    Document(
        page_content="""
MS Dhoni is one of the most recognizable players in Indian cricket and a highly successful IPL captain. He has represented Chennai Super Kings (CSK) and is known for his leadership, wicketkeeping skills, finishing ability, and calm decision-making under pressure. Dhoni has played an important role in the success and identity of Chennai Super Kings.
""",
        metadata={
            "player": "MS Dhoni",
            "team": "CSK",
            "role": "Wicketkeeper-Batter"
        }
    ),

    Document(
        page_content="""
Jasprit Bumrah is an Indian fast bowler and one of the most highly regarded bowlers in modern cricket. He has represented Mumbai Indians in the IPL. Bumrah is known for his unusual bowling action, accuracy, yorkers, slower deliveries, and ability to perform during the death overs. He is also capable of taking important wickets during difficult phases of a match.
""",
        metadata={
            "player": "Jasprit Bumrah",
            "team": "MI",
            "role": "Bowler"
        }
    ),

    Document(
        page_content="""
Ravindra Jadeja is an Indian all-rounder who has been an important player for Chennai Super Kings in the IPL. He is a left-handed batter and left-arm orthodox spin bowler. Jadeja is known for his all-round abilities, excellent fielding, bowling accuracy, and ability to contribute with the bat. His athletic fielding has made him one of the most valuable all-round players in T20 cricket.
""",
        metadata={
            "player": "Ravindra Jadeja",
            "team": "CSK",
            "role": "All-rounder"
        }
    )
]

# Create Chroma vector store
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="chroma_db",
    collection_name="sample"
)

# Add documents
vector_store.add_documents(documents)

#view documents
vector_store.get(include=['embedings','documents','metadata'])

#search documents
vector_store.similarity_search(
    query='who among this are bowler?',
    k=2
)

#search with similarity score
vector_store.similarity_search_with_score(
    query='who among this are bowler?',
    k=2
)

#metadata filtering
vector_store.similarity_search_with_score(
    query='who among this are bowler?',
    filter={"team":"chennai super Kings"}
)
