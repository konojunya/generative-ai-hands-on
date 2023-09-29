import os, time, json
import asyncio
import asyncpg
import numpy as np
from google.cloud import storage
from google.cloud.sql.connector import Connector
from cloudevents.http import from_http
from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.asyncpg import register_vector


# Parameters
project_id = os.environ.get("PJID", None)
region = "asia-northeast1"
instance_name = "pg15-pgvector-demo"
database_name = "docs"
database_user = "docs-admin"
database_password = "handson"


app = Flask(__name__)


async def insert_doc(file_id:int, text:str, metadata:str, embeddings_data:list):
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await register_vector(conn)
        
        await conn.execute(
            "INSERT INTO docs_embeddings (product_id, content, metadata, embedding) VALUES ($1, $2, $3, $4)",
            file_id,
            text,
            json.dumps(metadata),
            np.array(embeddings_data),
        )
        await conn.close()


async def search_doc(embeddings_data:list):
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await register_vector(conn)
        similarity_threshold = 0.0
        num_matches = 50

        # Find similar products to the query using cosine similarity search
        # over all vector embeddings. This new feature is provided by `pgvector`.
        results = await conn.fetch(
            """SELECT product_id, content, metadata, 1 - (embedding <=> $1) AS similarity
            FROM docs_embeddings
            WHERE 1 - (embedding <=> $1) > $2
            ORDER BY similarity DESC
            LIMIT $3
            """,
            embeddings_data,
            similarity_threshold,
            num_matches,
        )

        await conn.close()
        
        return results

    
def call_Palm(context:str, question:str):
    llm = VertexAI(
        model_name="text-bison@001",
        max_output_tokens=256,
        temperature=0.1,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )
    
    template = """
    ###{context}###
    ###で囲まれたテキストから、"質問：{question}" に関連する情報を抽出してください。
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    final_prompt = prompt.format(context=context, question=question)
    result = llm(final_prompt)

    return result


def download_from_gcs(bucket_name:str, name:str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(name)
    name = name.split("/")[-1]
    blob.download_to_filename(name)
    

@app.route("/")
def index():
    return "<p>This is Gen AI API</p>"


@app.post('/register_doc')
async def register_doc():
    """
    This handler is trigered from pubsub.
    """
    event = from_http(request.headers, request.data)
    data = event.data
    bucket_name = data["bucket"]
    name = data["name"]
    
    if not ".pdf" in name:
        return ("This is not pdf file", 200)
    
    # download pdf form gcs
    download_from_gcs(bucket_name, name)
    
    # read pdf
    name = name.split("/")[-1]
    loader = PyPDFLoader(name)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "。"],
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
    )
    pages = loader.load_and_split(text_splitter=text_splitter)
    
    # Create embeddings and inser data to Cloud SQL
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-multilingual@latest")
    for page in pages:
        embeddings_data = embeddings.embed_query(page.page_content)
        # Filtering data
        cc = page.page_content.encode("utf-8").replace(b'\x00', b'').decode("utf-8")
        await insert_doc(name, cc, page.metadata, embeddings_data)
        
    return ("Registered a doc in Cloud SQL", 200)


@app.post("/search")
async def search():
    """
    Doc search and call LLM with a prompt.
    """
    data = request.get_json()
    
    # Crate embeddings
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-multilingual@latest")
    embeddings_data = embeddings.embed_query(data["question"])
    
    # Serch docs
    results = await search_doc(embeddings_data)
    
    # Call LLM
    llm_result = call_Palm(results[0]["content"], data["question"])
    
    # Create a response
    response = {
        "answer": llm_result,
        "metadata": json.loads(results[0]["metadata"])
    }
    
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))
