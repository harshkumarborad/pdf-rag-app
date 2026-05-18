"""
test_suite.py — 5 Hard-Coded Test Questions per Domain
=======================================================
Provides 5 question sets × 5 domains (+ generic).

Each question has:
  - question: the query string
  - expected_keywords: terms we expect to appear in a good answer
  - category: human-readable label

Scoring per question:
  - retrieval_score:   avg cosine similarity of retrieved chunks
  - context_relevancy: how relevant chunks are to this question
  - faithfulness:      how grounded the answer is in context
  - answer_relevancy:  how directly the answer addresses the question
  - overall:           weighted average
  - keyword_hit_rate:  fraction of expected_keywords found in answer (custom metric)
"""

from typing import List, Dict, Any

# ── Question Banks ─────────────────────────────────────────────────────────────

QUESTION_SETS: Dict[str, List[Dict[str, Any]]] = {

    "Generic": [
        {
            "id": 1, "category": "Overview",
            "question": "What is the main topic or purpose of this document?",
            "expected_keywords": ["purpose", "scope", "objective", "main", "document", "about"],
        },
        {
            "id": 2, "category": "Definitions",
            "question": "What are the key definitions or important terms used in this document?",
            "expected_keywords": ["definition", "means", "refers", "defined", "term", "concept"],
        },
        {
            "id": 3, "category": "Requirements",
            "question": "What are the main requirements, obligations, or rules described?",
            "expected_keywords": ["shall", "must", "required", "obligation", "rule", "requirement"],
        },
        {
            "id": 4, "category": "Stakeholders",
            "question": "Who are the relevant parties, actors, or stakeholders mentioned?",
            "expected_keywords": ["party", "provider", "user", "operator", "authority", "stakeholder", "organisation"],
        },
        {
            "id": 5, "category": "Outcomes",
            "question": "What are the consequences, penalties, or next steps described in this document?",
            "expected_keywords": ["consequence", "penalty", "fine", "result", "next", "outcome", "enforcement"],
        },
    ],

    "Legal / Regulatory": [
        {
            "id": 1, "category": "Prohibited Acts",
            "question": "What practices, behaviours, or activities are explicitly prohibited under this regulation?",
            "expected_keywords": ["prohibited", "forbidden", "banned", "shall not", "not permitted", "illegal"],
        },
        {
            "id": 2, "category": "High-Risk Classification",
            "question": "How are high-risk categories or systems defined, and what criteria determine them?",
            "expected_keywords": ["high-risk", "risk", "category", "classification", "criteria", "annex"],
        },
        {
            "id": 3, "category": "Provider Obligations",
            "question": "What specific obligations apply to providers, deployers, or operators?",
            "expected_keywords": ["provider", "deployer", "operator", "obligation", "comply", "ensure", "register"],
        },
        {
            "id": 4, "category": "Transparency",
            "question": "What transparency, disclosure, or documentation requirements are mandated?",
            "expected_keywords": ["transparent", "disclosure", "inform", "document", "log", "notify", "label"],
        },
        {
            "id": 5, "category": "Enforcement & Penalties",
            "question": "What are the penalties, fines, and enforcement mechanisms for non-compliance?",
            "expected_keywords": ["penalty", "fine", "sanction", "enforcement", "violation", "million", "gdp"],
        },
    ],

    "Technical / Engineering": [
        {
            "id": 1, "category": "Specifications",
            "question": "What are the key technical specifications, performance requirements, or system parameters?",
            "expected_keywords": ["specification", "performance", "requirement", "parameter", "standard", "metric"],
        },
        {
            "id": 2, "category": "Architecture",
            "question": "What are the system components, modules, or architectural elements described?",
            "expected_keywords": ["component", "module", "architecture", "system", "interface", "layer", "design"],
        },
        {
            "id": 3, "category": "Testing & Validation",
            "question": "What testing, validation, or verification procedures are specified?",
            "expected_keywords": ["test", "validation", "verify", "procedure", "quality", "check", "accuracy"],
        },
        {
            "id": 4, "category": "Safety & Failure",
            "question": "What safety requirements, failure modes, or risk mitigations are described?",
            "expected_keywords": ["safety", "failure", "risk", "mitigation", "fault", "error", "redundancy"],
        },
        {
            "id": 5, "category": "Maintenance",
            "question": "What maintenance, upgrade, or lifecycle requirements are mentioned?",
            "expected_keywords": ["maintenance", "upgrade", "lifecycle", "update", "support", "version"],
        },
    ],

    "Business / Strategy": [
        {
            "id": 1, "category": "Objectives",
            "question": "What are the main business objectives, strategic goals, or mission described?",
            "expected_keywords": ["objective", "goal", "strategy", "mission", "vision", "aim", "target"],
        },
        {
            "id": 2, "category": "Market",
            "question": "What market segments, customer groups, or target audiences are identified?",
            "expected_keywords": ["market", "customer", "segment", "audience", "target", "client", "sector"],
        },
        {
            "id": 3, "category": "Competitive Advantage",
            "question": "What competitive advantages, unique selling points, or differentiators are described?",
            "expected_keywords": ["advantage", "unique", "differentiat", "competitive", "strength", "value"],
        },
        {
            "id": 4, "category": "KPIs & Financials",
            "question": "What financial metrics, KPIs, or success criteria are mentioned?",
            "expected_keywords": ["revenue", "profit", "kpi", "metric", "growth", "performance", "financial"],
        },
        {
            "id": 5, "category": "Risks",
            "question": "What risks, challenges, or mitigation strategies are identified?",
            "expected_keywords": ["risk", "challenge", "mitigation", "threat", "obstacle", "strategy"],
        },
    ],

    "HR / Recruitment": [
        {
            "id": 1, "category": "Qualifications",
            "question": "What qualifications, education, or experience are required for this role?",
            "expected_keywords": ["qualification", "experience", "degree", "education", "required", "skills", "years"],
        },
        {
            "id": 2, "category": "Responsibilities",
            "question": "What are the main responsibilities, duties, and day-to-day tasks described?",
            "expected_keywords": ["responsibility", "duty", "task", "role", "manage", "develop", "lead"],
        },
        {
            "id": 3, "category": "Benefits",
            "question": "What compensation, benefits, or working conditions are offered?",
            "expected_keywords": ["salary", "benefit", "compensation", "flexible", "remote", "leave", "insurance"],
        },
        {
            "id": 4, "category": "Culture",
            "question": "What company culture, values, or working environment are described?",
            "expected_keywords": ["culture", "value", "team", "environment", "diverse", "inclusive", "collaborative"],
        },
        {
            "id": 5, "category": "Application",
            "question": "What are the application process, selection criteria, or next steps?",
            "expected_keywords": ["apply", "application", "interview", "selection", "process", "contact", "send"],
        },
    ],
}


def compute_keyword_hit_rate(answer: str, expected_keywords: List[str]) -> float:
    """
    Fraction of expected_keywords (case-insensitive substring match) found in the answer.
    This is the custom retrieval score metric shown in the UI.
    """
    if not expected_keywords or not answer:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(hits / len(expected_keywords), 4)


def run_test_suite(
    question_set: str,
    session_id: str,
    top_k: int = 5,
    use_reranking: bool = False,
    llm_model: str = None,
) -> Dict[str, Any]:
    """
    Run all 5 questions from the selected set through the full pipeline.

    Args:
        question_set: One of the keys in QUESTION_SETS
        session_id:   Which ChromaDB collection to query
        top_k:        Chunks per question
        use_reranking: Enable cross-encoder re-ranking

    Returns:
        {
          "question_set": str,
          "results": [...per question...],
          "summary": {avg scores}
        }
    """
    from pipeline.retriever import retrieve
    from pipeline.generator import generate
    from pipeline.evaluator import evaluate

    questions = QUESTION_SETS.get(question_set, QUESTION_SETS["Generic"])
    results = []

    for q in questions:
        print(f"[TestSuite] Running Q{q['id']}: {q['question'][:60]}...")
        try:
            chunks = retrieve(q["question"], session_id, top_k=top_k, use_reranking=use_reranking)
            answer = generate(q["question"], chunks, model=llm_model)
            metrics = evaluate(q["question"], answer, chunks)
            khr = compute_keyword_hit_rate(answer, q["expected_keywords"])

            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": answer,
                "expected_keywords": q["expected_keywords"],
                "keyword_hit_rate": khr,
                "metrics": metrics,
                "sources": [
                    {
                        "file": c["metadata"].get("source_file", "?"),
                        "page": c["metadata"].get("page", "?"),
                        "excerpt": c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
                        "score": c["score"],
                    }
                    for c in chunks[:3]
                ],
            })
        except Exception as e:
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": f"Error: {e}",
                "expected_keywords": q["expected_keywords"],
                "keyword_hit_rate": 0.0,
                "metrics": {},
                "sources": [],
            })

    # Compute summary averages
    valid = [r for r in results if r["metrics"]]
    summary = {}
    if valid:
        for key in ["context_relevancy", "faithfulness", "answer_relevancy", "overall_score"]:
            summary[key] = round(sum(r["metrics"].get(key, 0) for r in valid) / len(valid), 4)
        summary["avg_keyword_hit_rate"] = round(
            sum(r["keyword_hit_rate"] for r in valid) / len(valid), 4
        )

    return {
        "question_set": question_set,
        "session_id": session_id,
        "results": results,
        "summary": summary,
    }
