# main.py

from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import loadvectorize
# import llmperfmonitor
# import timeit


def run_rag_qa():
    llm = LlamaCpp(
        model_path="mistral-7b-instruct-v0.1.Q2_K.gguf",
        temperature=0.01,
        max_tokens=2000,
        top_p=1,
        verbose=False,
        n_ctx=2048
    )

    # load document, vectorize and create retrievers
    db, bm25_r = loadvectorize.load_db("https://arxiv.org/pdf/2401.08406.pdf", "./arxiv_test_index")
    faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}, max_tokens_limit=1000)

    # instantiate an ensemble retriever
    r = 0.5
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_r, faiss_retriever], weights=[r, 1-r])

    # Prompt template
    qa_template = """<s>[INST] You are a helpful assistant.
    Use the following context to answer the question below comprehensively:
    {context}
    [/INST] </s>{question}
    """
    QA_PROMPT = PromptTemplate.from_template(qa_template)

    # create a retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=ensemble_retriever,
            chain_type_kwargs={"prompt": QA_PROMPT}
    )

    print("invoking")
    # print(llm.invoke("What is better, RAG or fine-tuning? Explain why."))
    print(llm.invoke("What does Shrike do?"))

    # ask query
    # query = "What is the purpose of a peering rule on SteelHead?"
    # result = qa_chain({"query": query})
    # print(f'Q: {query}\nA: {result["result"]}')

    query = "What is better, RAG or fine-tuning?"
    result = qa_chain({"query": query})
    print(f'Q: {query}\nA: {result["result"]}')

# print('model;question;cosine;meteor;exec_time')
# # weight ratio, r in [0.0, 0.25, 0.5, 0.75, 1.0]
# for r in [x/100 for x in range(0,101,25)]:
#     ensemble_retriever = EnsembleRetriever(retrievers=[bm25_r,faiss_retriever],weights=[r,1-r])
#     # QA Chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         retriever=ensemble_retriever,
#         chain_type_kwargs={"prompt": QA_PROMPT}
#     )

#     qa_list = llmperfmonitor.get_questions_answers()
#     # iterate through a list of questions, qa_list
#     for i,query in enumerate(qa_list[::2]):
#         start = timeit.default_timer()
#         result = qa_chain({"query": query})
#         # compute cosine similarity, meteor score and execution time
#         cos_sim = llmperfmonitor.calc_similarity(qa_list[i*2+1],result["result"])
#         meteor = llmperfmonitor.calculate_meteor_score(qa_list[i*2+1],result["result"])
#         time = timeit.default_timer() - start # seconds
#         # each output starts with the model name in form bm25-<r>_f-<1-r>
#         print(f'bm25-{r:.1f}_f-{11-r:.1f};{i+1};{cos_sim:.5};{meteor:.5};{time:.2f}')


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("--llm", type=str, required=True, help="URL to the PDF file")
    # argparser.add_argument("--doc_type", type=str, required=True, help="Path to the index file") pdf or website
    # argparser.add_argument("doc path")
    # args = argparser.parse_args()
    # args


    run_rag_qa()
