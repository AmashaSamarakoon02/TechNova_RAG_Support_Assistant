"""
evaluate.py — TechNova Support RAG Evaluation (standalone script)
=================================================================
Reads  : evaluation/questions.json   (official file)
Writes : eval_results.json

Run from the project root:
    python evaluate.py
"""
import json, re, time, pathlib
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(override=True)

MODEL     = "gpt-4.1-nano"
EMBED     = "all-MiniLM-L6-v2"
DB_NAME   = "vector_db"
KB_PATH   = pathlib.Path("knowledge-base")
EVAL_FILE = pathlib.Path("evaluation/questions.json")
OUTPUT    = pathlib.Path("eval_results.json")
TOP_K     = 5

SYSTEM_PROMPT = """You are a professional support assistant for TechNova.
Answer ONLY using the CONTEXT chunks provided — never use outside knowledge.
After every factual statement cite its source: (source: path/filename.md)
If the context does NOT contain the answer, say exactly:
"I don't have that information in my knowledge base."
NEVER fabricate a source not listed in the context.
CONTEXT:\n{context}"""

JUDGE_PROMPT = """Strict exam grader.
Question       : {question}
Expected facts : {expected}
Student answer : {answer}
Score 0-5 (5=all facts correct, 0=wrong). JSON only: {{"score":<int>,"reason":"<one sentence>"}}"""


def load_vectorstore():
    if not pathlib.Path(DB_NAME).exists():
        raise FileNotFoundError(
            f"'{DB_NAME}' not found. Run Task2_VectorStore.ipynb first to build it."
        )
    emb = HuggingFaceEmbeddings(model_name=EMBED)
    vs  = Chroma(persist_directory=DB_NAME, embedding_function=emb)
    print(f"Vector store loaded: {vs._collection.count()} chunks")
    return vs


def retrieve(vs, question, k=TOP_K, category=None):
    kw = {"k": k}
    if category and category not in (None, "all", "All"):
        kw["filter"] = {"category": category}
    return vs.as_retriever(search_kwargs=kw).invoke(question)


def answer(llm, vs, question):
    docs = retrieve(vs, question)
    ctx  = "\n\n".join(
        f"[{i}] Extract from {d.metadata.get('source','?')}:\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )
    msgs = [SystemMessage(content=SYSTEM_PROMPT.format(context=ctx)),
            HumanMessage(content=question)]
    return llm.invoke(msgs).content, docs


def kw_score(answer, must):
    if not must: return 1.0
    lower = answer.lower()
    return round(sum(1 for k in must if k.lower() in lower) / len(must), 3)


def llm_judge(llm, question, must, answer_text):
    p = JUDGE_PROMPT.format(
        question=question,
        expected=", ".join(must),
        answer=answer_text[:800]
    )
    try:
        raw  = llm.invoke([HumanMessage(content=p)]).content
        data = json.loads(re.sub(r"```json|```", "", raw).strip())
        return {"score": data.get("score", 0) / 5, "reason": data.get("reason", "")}
    except Exception as e:
        return {"score": 0.0, "reason": str(e)}


def ret_hit(docs, expected):
    if not expected: return True
    rs = {d.metadata.get("source", "") for d in docs}
    return any(e in s for e in expected for s in rs)


def is_refusal(text):
    return any(p in text.lower() for p in
               ["don't know", "not have", "cannot", "don't have that information"])


def main():
    print("=" * 55)
    print("TechNova Support — RAG Evaluation")
    print("=" * 55)

    # ── Check files exist ──────────────────────────────────────────────
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"'{EVAL_FILE}' not found.\n"
            "Run from the project root folder containing the evaluation/ subfolder."
        )

    # ── Load resources ─────────────────────────────────────────────────
    vs  = load_vectorstore()
    llm = ChatOpenAI(model=MODEL, temperature=0)

    with open(EVAL_FILE, encoding="utf-8") as f:
        eval_questions = json.load(f)["questions"]
    print(f"Loaded {len(eval_questions)} questions from {EVAL_FILE}\n")

    # ── Main loop ──────────────────────────────────────────────────────
    per_q     = []
    kw_scores = []
    jg_scores = []
    ret_hits  = 0
    isc       = 0
    oos_ok    = None

    for q in eval_questions:
        print(f"  {q['id']}...", end=" ", flush=True)

        answer_text, docs = answer(llm, vs, q["question"])
        ks  = kw_score(answer_text, q.get("must_mention", []))
        jg  = llm_judge(llm, q["question"], q.get("must_mention", []), answer_text)
        hit = ret_hit(docs, q.get("expected_sources", []))

        if q["category"] != "out_of_scope":
            isc      += 1
            ret_hits += int(hit)
        else:
            refused = is_refusal(answer_text)
            oos_ok  = refused
            ks      = 1.0 if refused else 0.0
            jg["score"] = ks
            print(f"REFUSAL={'OK' if refused else 'FAIL'}  ", end="")

        kw_scores.append(ks)
        jg_scores.append(jg["score"])

        per_q.append({
            "id"               : q["id"],
            "category"         : q["category"],
            "question"         : q["question"],
            "predicted_answer" : answer_text,
            "retrieved_sources": [d.metadata.get("source") for d in docs],
            "retrieval_hit"    : hit,
            "keyword_score"    : ks,
            "passed"           : ks >= 0.5,
            "llm_score"        : jg["score"],
            "llm_reason"       : jg["reason"],
        })
        print(f"kw={ks:.2f}  llm={jg['score']:.2f}")
        time.sleep(0.3)

    # ── Summary ────────────────────────────────────────────────────────
    summary = {
        "total_questions"    : len(eval_questions),
        "passed"             : sum(1 for r in per_q if r["passed"]),
        "failed"             : sum(1 for r in per_q if not r["passed"]),
        "retrieval_precision": round(ret_hits / isc, 3) if isc else 0,
        "keyword_score_avg"  : round(sum(kw_scores) / len(kw_scores), 3),
        "llm_score_avg"      : round(sum(jg_scores) / len(jg_scores), 3),
        "refused_correctly"  : bool(oos_ok),
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_q}, f, indent=2)

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print(f"  Total questions     : {summary['total_questions']}")
    print(f"  Passed              : {summary['passed']} ({summary['passed']/summary['total_questions']:.0%})")
    print(f"  Failed              : {summary['failed']}")
    print(f"  Retrieval precision : {summary['retrieval_precision']:.0%}")
    print(f"  Keyword score avg   : {summary['keyword_score_avg']:.0%}")
    print(f"  LLM judge score avg : {summary['llm_score_avg']:.0%}")
    print(f"  Out-of-scope refused: {summary['refused_correctly']}")
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
