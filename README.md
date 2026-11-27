# FTEC5660 Homework 4

This project retrieves evidence from annual reports, earnings calls, and broker research reports for a fixed question answering model.

## Run locally

Clone the repository and enter its directory. Use Python 3.10 or newer.

~~~sh
git clone https://github.com/Enterprise2333/FTEC5660-HW4.git
cd FTEC5660-HW4
python -m venv .venv
~~~

Activate the virtual environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in macOS or Linux. Set `DEEPSEEK_API_KEY` in your environment or create a local `.env` file with `DEEPSEEK_API_KEY=your_key_here` and replace the placeholder with your own key.

~~~sh
python -m pip install -r requirements.txt
python hw4.py --data-folder public_test
~~~

The runner rebuilds the in-memory index and writes `results.csv`. To reproduce the ablation, run `python run_ablation.py --data-folder public_test`. Each configuration writes its own CSV and console log under `ablation/`. Use `--variant V6` to run just one configuration.

## Homework 4 solution

### Pipeline

~~~mermaid
flowchart LR
    A[HTML reports] --> B[Read metadata and label table cells]
    B --> C[Split at section and table row boundaries]
    C --> D[Build BM25 index in memory]
    Q[Question] --> E[Extract company, year, and report type]
    E --> F[Filter and score evidence]
    D --> F
    F --> G{Evidence available?}
    G -->|Yes| H[Select up to five chunks]
    H --> I[Fixed DeepSeek generator]
    G -->|No| J[NOT_FOUND]
~~~

The parser keeps each financial value with its row label, year column, and table caption. Section based chunks keep related text together within the 1,500 character limit. Each chunk retains its company, document type, fiscal year, and source. The retriever combines BM25 with company and document filters and gives more weight to matching metric rows and years. It checks annual report comparison columns and forecast periods separately from the publication year. Comparison questions reserve evidence for each company. The retriever returns an empty list when the company or year is unavailable or when the required disclosure terms are absent. All retrieval steps are lexical and use zero model calls.

### Public ablation

Every row used the same 40 documents, 60 questions, and course generator. Recall and kept use the 46 answerable questions. Rejected uses the 14 unanswerable questions and counts only empty retrievals. Time includes indexing, retrieval, and generation. Tokens come from the course generator's tracker.

The unchanged course starter answered all questions with NOT_FOUND. Its separate run scored 4.67/75 and answered 14/60 correctly. B0 is the first retrieval baseline.

| Configuration | Change from previous row | Recall@5 | Comparison recall | Kept | Rejected | Answer accuracy | Generator tokens, input / output | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | Plain HTML text, recursive splitting, whole corpus BM25 | 17/46 | 2/12 | 46/46 | 0/14 | 28/60 | 124,013 / 180 | 4.6 s |
| V1 | Label table cells with row and column | 14/46 | 2/12 | 46/46 | 0/14 | 27/60 | 123,889 / 182 | 5.4 s |
| V2 | Use section and table row boundaries | 11/46 | 0/12 | 46/46 | 0/14 | 24/60 | 63,926 / 179 | 5.7 s |
| V3 | Filter candidates with parsed company, year, and document type | 33/46 | 5/12 | 46/46 | 0/14 | 47/60 | 67,729 / 202 | 5.2 s |
| V4 | Boost matching metric rows and periods | 46/46 | 12/12 | 46/46 | 0/14 | 60/60 | 87,825 / 203 | 5.9 s |
| V5 | Reserve a comparison chunk per company | 46/46 | 12/12 | 46/46 | 0/14 | 60/60 | 87,825 / 203 | 5.8 s |
| V6 | Reject queries when the required evidence is unavailable | 46/46 | 12/12 | 46/46 | 14/14 | 60/60 | 64,970 / 163 | 5.1 s |

V1 makes row and year associations explicit, but changes the tokens and their distribution in a global BM25 index. Measured recall fell by three questions. V2 produced shorter evidence bundles and cut generator input to roughly half of V1. Under unrestricted BM25, narrower chunks carried fewer company and topic terms together. V3 gained 22 recall hits by restricting the search to relevant documents. V4 recovered the remaining 13 hits by favoring exact labels and reported periods. V5 preserved perfect public comparison recall while reserving a passage for each company by design. V6 saved 14 generator calls and approximately 26% of V5's input tokens. The generator had already answered those 14 questions correctly at V5.

### Error analysis

- **Company identity, q001.** B0 answered 241.6 for Northgate Robotic Systems, a company absent from the corpus. The retrieved evidence referred to Northgate Robotics. Exact company matching in V6 caused the retriever to return no chunks.
- **Exact figure versus rounded narrative, q002.** V2 answered 1,600. The expected reported revenue was 1,574.7. The final ranking prioritizes the labelled revenue row over narrative approximations.
- **Metric and period, q016.** V3 returned a calculation using rounded figures for FY2024 gross profit. The fixed scorer needs one number, and the printed amount is 1,805.4. V4 retrieved the labelled gross profit row with the requested year and the generator returned the printed value.
- **Two facts for one answer, q021.** B0 missed the cash comparison. V6 provides both companies' 2024 cash figures, 215.7 and 114.5, so the generator can choose the company with more cash.

V6 is the default configuration and corresponds to `results.csv`. It scored 75/75 on the public automatic rubric. All 12 tables, 40 documents, and 180 query fields passed. These measurements use the public corpus. New wording can require lexical matches outside the current metric and company rules.

### Early ablation improvements

These runs use the same 60 public questions, fixed generator, and five-chunk limit as the original ablation. The three baseline reruns reproduced every recorded answer correctness and evidence hit. The company option prepends the company from each chunk's metadata to the text fed to BM25. The normalized option strips possessives, splits fiscal-year prefixes and punctuation, and keeps decimal and grouped numbers intact. The generator receives the original chunks in every run.

| Base | Index option | Recall@5 | Table recall | Fact recall | Comparison recall | Correct answers | Gains / losses vs base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | Original | 17/46 | 5/22 | 10/12 | 2/12 | 28/60 | — |
| B0 | Company | 28/46 | 12/22 | 11/12 | 5/12 | 41/60 | 14 / 1 |
| B0 | Normalized | 19/46 | 7/22 | 9/12 | 3/12 | 29/60 | 5 / 4 |
| B0 | Company + normalized | 30/46 | 13/22 | 11/12 | 6/12 | 43/60 | 15 / 0 |
| V1 | Original | 14/46 | 3/22 | 9/12 | 2/12 | 27/60 | — |
| V1 | Company | 25/46 | 10/22 | 12/12 | 3/12 | 38/60 | 12 / 1 |
| V1 | Normalized | 17/46 | 6/22 | 8/12 | 3/12 | 31/60 | 6 / 2 |
| V1 | Company + normalized | 32/46 | 16/22 | 10/12 | 6/12 | 45/60 | 20 / 2 |
| V2 | Original | 11/46 | 4/22 | 7/12 | 0/12 | 24/60 | — |
| V2 | Company | 36/46 | 16/22 | 9/12 | 11/12 | 50/60 | 27 / 1 |
| V2 | Normalized | 12/46 | 5/22 | 7/12 | 0/12 | 26/60 | 2 / 0 |
| V2 | Company + normalized | 37/46 | 19/22 | 10/12 | 8/12 | 50/60 | 28 / 2 |
| V2 | Company + evidence ranking | 45/46 | 22/22 | 11/12 | 12/12 | 55/60 | 33 / 2 |
| V2 | Company + normalized + evidence ranking | 45/46 | 22/22 | 11/12 | 12/12 | 56/60 | 34 / 2 |

The company index delivered V2's largest gain, adding 26 correct answers. Normalizing that index recovered three more table evidence hits and lost three comparison evidence hits. Evidence ranking reuses V4's metric and period weights on chunks belonging to the companies parsed from the question. It keeps the entire corpus eligible and yields **56/60** with both index options active. The V6 regression run scored **75/75** and answered **60/60**.

- In q002, V2 answered with the rounded amount 1,600. With the company index alone, the first chunk containing 1,574.7 ranked tenth. Normalizing tokens moved that value into the first result, and the generator returned 1,574.7.
- In the strongest V2 experiment, q027 missed the headquarters evidence for a question without a fiscal year. q035 returned NOT_FOUND despite retrieving the target price, and q050 returned the Americas figure despite retrieving the requested regional row. q044 retrieved data for a similarly named company and returned an answer for a company absent from the corpus.

[Per-question CSVs and runtime logs](ablation/early/) contain the full results. Reproduce the three index experiments across B0, V1, and V2 with the following commands. For an original baseline, use `--index original --variant B0`, substituting V1 or V2 as needed.

~~~sh
python run_ablation.py --index company --output-dir ablation/early
python run_ablation.py --index normalized --output-dir ablation/early
python run_ablation.py --index company-normalized --output-dir ablation/early
python run_ablation.py --variant V2 --index company --evidence-rank --output-dir ablation/early
python run_ablation.py --variant V2 --index company-normalized --evidence-rank --output-dir ablation/early
~~~

[Task 6 reflection](output/pdf/HW4_Reflection.pdf)
