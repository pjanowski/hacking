import argparse
import os

from bs4 import BeautifulSoup as Soup
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    RecursiveUrlLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings

from get_web_subpages import get_web_subpages


def model_llamacpp(temperature):
    llm = LlamaCpp(
        model_path="../rag/mistral-7b-instruct-v0.1.Q2_K.gguf",
        temperature=temperature,
        max_tokens=2000,
        verbose=False,
        n_ctx=2048,
    )
    print("Using model: mistral-7b-instruct on CPU.")
    return llm


def model_gpt35T(temperature):
    llm = AzureOpenAI(
        deployment_name="35Tinstruct",
        temperature=temperature,
        max_tokens=2000,
        n=1,
        verbose=False,
    )
    print("Model: GPT-35-Turbo on Azure.")
    return llm


def model_gpt4(temperature):
    llm = AzureChatOpenAI(
        deployment_name="40125",
        temperature=temperature,
        max_tokens=2000,
        verbose=False,
    )
    print("Model: GPT-4-0125 on Azure.")
    return llm


def model_gpt4mini(temperature):
    #   self.tokenProvider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    llm = AzureChatOpenAI(
        deployment_name="mini",
        temperature=temperature,
        max_tokens=2000,
        verbose=False,
    )
    print("Model: GPT-4o-mini on Azure.")
    return llm


def get_retriever(docloader, r):
    # embeddings = AzureOpenAIEmbeddings(azure_deployment="ada3large")
    embeddings = HuggingFaceEmbeddings()  # (uses latest sentence transformers model)

    if docloader == "recursiveurl":
        index_path = "./recursiveurlloader_shrike_index"
        docs_func = retriever_recursiveurl
    elif docloader == "webbase":
        index_path = "./webbaseloader_shrike_index"
        docs_func = retriever_webbaseloader
    elif docloader == "markdown":
        index_path = "./markdownloader_shrike_index"
        docs_func = retriever_markdown
    else:
        raise ValueError(f"Docloader {docloader} not supported")

    # load and split docs if necessary
    if not os.path.exists(index_path) or r != 1:
        print("Loading docs.")
        docs = docs_func(index_path)

        if not os.path.exists(index_path):
            text_splitter = SemanticChunker(embeddings)
        elif r != 1:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
            # text_splitter = SemanticChunker(HuggingFaceEmbeddings())

        docs = text_splitter.split_documents(docs)

    # create retrievers
    if r != 0:
        if not os.path.exists(index_path):
            print("Building index.")
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(index_path)
            print("Index created and saved to disk.")
        else:
            print("Index exists, loading.")
            db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 5}, max_tokens_limit=2000)

    if r != 1:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3

    # return retriever
    if r == 1:
        print("Using full FAISS retriever.")
        return faiss_retriever
    elif r == 0:
        print("Using full BM25 retriever.")
        return bm25_retriever
    else:
        print(f"Using ensemble retriever with ratio {r}.")
        return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[r, 1 - r])


def retriever_recursiveurl(index_path):
    url = "https://azure.github.io/shrike/"
    loader = RecursiveUrlLoader(url=url, extractor=lambda x: Soup(x, "html.parser").text)
    docs = loader.load()
    return docs


def retriever_webbaseloader(index_path):
    url = "https://azure.github.io/shrike/"
    urls = get_web_subpages(url)
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs


def retriever_markdown(index_path):
    markdown_path = "../../shrike/"
    # TODO: load python files with PythonLoader,
    # https://python.langchain.com/v0.1/docs/integrations/document_loaders/source_code/,
    # https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/file_directory/
    loader = DirectoryLoader(markdown_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, silent_errors=True)
    markdown_raw = loader.load()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    markdown_splits = []
    for doc in markdown_raw:
        markdown_splits.extend(markdown_splitter.split_text(doc.page_content))

    return markdown_splits


def main(user_prompt, model, docloader, noRAG, temperature, r):
    # model
    if model == "llamacpp":
        llm = model_llamacpp(temperature)
    elif model == "gpt35T":
        llm = model_gpt35T(temperature)
    elif model == "gpt4":
        llm = model_gpt4(temperature)
    else:
        raise ValueError(f"Model {model} not supported")

    # no RAG
    if noRAG:
        print("Using LLM directly")
        # invoke is simple. Generate provides all details including token usage and multiple responses if n>1.
        # Callback for openai provides cost.
        response = llm.invoke(user_prompt)
        response = StrOutputParser().invoke(response)
        return response

    # Data retrieval
    retriever = get_retriever(docloader, r)

    # Prompt template
    template = """You are a helpful assistant. Use the following context to answer the question at the end.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Answer the question comprehensively.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""
    # prompt = PromptTemplate.from_template(template)
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    setup_and_retrieval = {"context": retriever | format_docs, "question": RunnablePassthrough()}
    output_parser = StrOutputParser()
    rag_chain = setup_and_retrieval | prompt | llm | output_parser

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) | prompt | llm | output_parser
    )

    retrieve_docs = (lambda x: x["question"]) | retriever

    rag_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(answer=rag_chain_from_docs)

    # TODO: authenticate with Azure MSI
    print("Invoking chain")
    result = rag_chain.invoke({"question": user_prompt})
    # result = rag_chain.invoke("What are the major features of Shrike?")

    return result


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="What does Shrike do?",
        help="User prompt to use for query.",
    )
    argparser.add_argument(
        "--model", "-m", type=str, default="llamacpp", choices=["llamacpp", "gpt35T", "gpt4"], help="LLM model to use"
    )
    argparser.add_argument(
        "--docloader",
        "-d",
        type=str,
        default="recursiveurl",
        choices=["recursiveurl", "webbase", "markdown"],
        help="Langchain document loader to use",
    )
    argparser.add_argument("--noRAG", action="store_true", help="Use RAG retrieval. If false query model directly.")
    argparser.add_argument("--temperature", "-t", type=float, default=0.2, help="Temperature for LLM model")
    argparser.add_argument(
        "--retriever_weight_ratio",
        "-r",
        type=float,
        default=1,
        help="Weight ratio for retrievers. 1=full FAISS, 0=full BM25. Default=1.",
    )
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    response = main(args.prompt, args.model, args.docloader, args.noRAG, args.temperature, args.retriever_weight_ratio)
    print(f"Q: {args.prompt}")
    print(f"A: {response}")
