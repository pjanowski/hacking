import loadvectorize
import main

test_fileurl = "https://arxiv.org/pdf/2401.08406.pdf"
test_index_path = "./arxiv_test_index"


def test_helloworld():
    hw = "Hello, World!"
    assert hw == "Hello, World!"


def test_load_doc(fileurl=test_fileurl):
    docs = loadvectorize.load_doc(fileurl)
    print(f"Number of documents: {len(docs)}")

    import pickle

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    assert len(docs) > 1


def test_vectorize():
    docs = loadvectorize.load_doc(test_fileurl)
    # db,bm25_retriever = loadvectorize.vectorize(docs=docs, index_path=test_index_path, force_rebuild=True)
    db, bm25_retriever = loadvectorize.vectorize(docs=docs, index_path=test_index_path, force_rebuild=False)
    assert db is not None
    assert bm25_retriever is not None


def test_load_db():
    db, bm25_retriever = loadvectorize.load_db(test_fileurl, test_index_path)
    assert db is not None
    assert bm25_retriever is not None


def test_main():
    main.run_rag_qa()
    assert True
