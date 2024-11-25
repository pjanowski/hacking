from sentence_transformers import SentenceTransformer, util
from nltk.translate import meteor
from nltk import word_tokenize


# returns a list of questions interleaved with answers
def get_questions_answers():
    with open("sh_qa_list.txt") as qfile:
        lines = [line.rstrip()[3:] for line in qfile]
    return lines


# calculates cosine between two sentences
def calc_similarity(sent1, sent2):
    # creates embeddings, computes cosine similarity and returns the value
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # compute embedding for both strings
    embedding_1 = model.encode(sent1, convert_to_tensor=True)
    embedding_2 = model.encode(sent2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2).item()


# calculates meteor score between two sentences
def calculate_meteor_score(response, ref):
    reference = ref
    hypothesis = response

    score = meteor(
        [word_tokenize(reference)],
        word_tokenize(hypothesis)
    )
    return score
