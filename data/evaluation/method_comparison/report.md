# Retrieval method comparison
Generated 2026-11-14 18:58. k = 4 (the number of chunks given to the tutor). Test sets: A = 22 annotated queries, B = 455 quiz bank (weak labels), C = 35 student-style questions.

## 1. Results

All searches without topic information. Higher is better for every column except time.

| method | A recall@4 | A hit@4 | A MRR | B hit@4 | B precision@4 | B MRR | C precision@4 | C hit@4 | median ms |
|---|---|---|---|---|---|---|---|---|---|
| tfidf | 0.634 | 0.909 | 0.729 | 0.585 | 0.259 | 0.481 | 0.671 | 0.914 | 0.2 |
| bm25 | 0.695 | 0.864 | 0.752 | 0.604 | 0.273 | 0.473 | 0.671 | 0.914 | 0.3 |
| dense (Qwen3-Embedding-0.6B) | 0.562 | 0.864 | 0.782 | 0.842 | 0.478 | 0.709 | 0.893 | 1.000 | 21.8 |
| hybrid (Qwen3-Embedding-0.6B) | 0.802 | 0.955 | 0.886 | 0.798 | 0.425 | 0.661 | 0.843 | 0.971 | 22.2 |

## 2. Difference from BM25 (95% bootstrap interval)

Each cell is the mean difference from BM25 on one primary measure, the interval, and a verdict. 'better' or 'worse' only when the interval excludes zero. Set A has few queries, so its intervals are wide.

| method | A recall@4 | B MRR | C precision@4 |
|---|---|---|---|
| tfidf | -0.061 [-0.220, +0.102] no reliable difference | +0.007 [-0.025, +0.040] no reliable difference | +0.000 [-0.064, +0.064] no reliable difference |
| dense (Qwen3-Embedding-0.6B) | -0.133 [-0.341, +0.079] no reliable difference | +0.236 [+0.196, +0.279] better | +0.221 [+0.143, +0.307] better |
| hybrid (Qwen3-Embedding-0.6B) | +0.108 [-0.023, +0.253] no reliable difference | +0.188 [+0.158, +0.219] better | +0.171 [+0.093, +0.257] better |

## 3. Using the orchestrator's topic as a soft preference (student-style questions)

Share of the retrieved chunks that are on-topic. This is how the tutor calls the retriever.

| method | no topic | topic right (soft boost) | topic wrong on 1 in 5 |
|---|---|---|---|
| tfidf | 67% | 93% | 86% |
| bm25 | 67% | 91% | 84% |
| dense (Qwen3-Embedding-0.6B) | 89% | 100% | 85% |
| hybrid (Qwen3-Embedding-0.6B) | 84% | 100% | 88% |

## 4. Qwen3 query instruction

Only the question is changed; the chunks are embedded without an instruction. The first variant is the one used in the tables above and was chosen before running. Picking the best of the three on this data would be slightly optimistic.

| instruction | method | A recall | B rr | C precision |
|---|---|---|---|---|
| tutoring instruction (used above) | dense (Qwen3-Embedding-0.6B) | 0.562 | 0.709 | 0.893 |
| tutoring instruction (used above) | hybrid (Qwen3-Embedding-0.6B) | 0.802 | 0.661 | 0.843 |
| built-in 'query' prompt | dense (Qwen3-Embedding-0.6B) | 0.582 | 0.653 | 0.886 |
| built-in 'query' prompt | hybrid (Qwen3-Embedding-0.6B) | 0.810 | 0.626 | 0.829 |
| no instruction | dense (Qwen3-Embedding-0.6B) | 0.573 | 0.648 | 0.879 |
| no instruction | hybrid (Qwen3-Embedding-0.6B) | 0.791 | 0.636 | 0.843 |

## 5. Recommendation

**Use hybrid (Qwen3-Embedding-0.6B).** It is reliably better than BM25 on 2 of 3 primary measures and reliably worse on none. Its median search time is 22.2 ms against 0.3 ms for BM25, and it needs the embedding model loaded next to the tutor model.

## How to read this

- Differences smaller than the intervals are noise. With 22 annotated queries one query moves a score by about 4 points.
- Set B labels are weak (sharing a subtopic does not prove a chunk answers the question), so use it to compare methods, not as an absolute score.
- Dense and hybrid need the embedding model next to the tutor model on the GPU, and a search takes longer than BM25.
