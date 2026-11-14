# RAG evaluation report
Generated 2026-11-14 17:42. Model sections not run (use --llm).

## 1. Retrieval relevance

22 annotated queries with known relevant chunks. Recall@5 = share of the relevant chunks found in the top 5; MRR = 1/rank of the first relevant chunk; hit@5 = share of queries with at least one relevant chunk in the top 5.

| method | queries | Recall@5 | MRR | hit@5 | median ms |
|---|---|---|---|---|---|
| tfidf | 22 | 0.748 | 0.729 | 0.955 | 0.27 |
| bm25 | 22 | 0.787 | 0.752 | 0.909 | 0.32 |

Lexical ablation (BM25):

| stopword removal | stemming | Recall@5 | MRR | hit@5 |
|---|---|---|---|---|
| off | off | 0.783 | 0.754 | 0.909 |
| off | on | 0.734 | 0.797 | 0.909 |
| on | off | 0.787 | 0.752 | 0.909 |
| on | on | 0.749 | 0.795 | 0.909 |

## 2. Topic handling

35 student-style questions, top 4 chunks. 'On-topic' = the chunk's topic is one of the topics a correct answer should come from. Topic judged, not exact chunk.

| how the topic is used | on-topic (topic right) | questions with an on-topic chunk | on-topic (topic wrong on 1 in 5) | questions with an on-topic chunk (1 in 5 wrong) |
|---|---|---|---|---|
| no topic information | 67% | 32/35 | 67% | 32/35 |
| topic name added to the query | 85% | 34/35 | 81% | 33/35 |
| hard filter (topic_ids) | 100% | 35/35 | 80% | 28/35 |
| soft boost, x2 (preferred_topics) | 91% | 35/35 | 84% | 34/35 |

## 3. Corpus safety checks

57 queries, top 4 chunks, beginner profile (mastery 0.3 in every topic).

| safety checks | chunks | use later topics | duplicates | fragments without background |
|---|---|---|---|---|
| off | 228 | 16 | 3 | 37 |
| on | 228 | 0 | 0 | 0 |

## 4. Personalisation by mastery

Does the retrieved material get harder for stronger students? Difficulty scale 1-3; right topic given as a preference.

| difficulty weight | mean difficulty, weak 0.2 | on-topic, weak 0.2 | mean difficulty, developing 0.55 | on-topic, developing 0.55 | mean difficulty, strong 0.9 | on-topic, strong 0.9 | gap strong-weak |
|---|---|---|---|---|---|---|---|
| 0.0 | 2.01 | 91% | 2.01 | 91% | 2.01 | 91% | +0.00 |
| 0.5 | 1.81 | 91% | 1.96 | 91% | 2.22 | 89% | +0.41 |
| 1.0 | 1.76 | 90% | 1.99 | 90% | 2.33 | 86% | +0.56 |
