"""
Prompt templates for G-Eval detection.
"""

FAITH_PROMPT_TEMPLATE = '''G-Eval for Evaluating Faithfulness

### Task Introduction:
Given a complex question and a generated answer about a table, your task is to rate the answer's Faithfulness.

### Evaluation Criteria:
Faithfulness (1-5): A good answer should accurately and completely address the given question. It must be based entirely on the information provided and should not include any unfaithful or hallucinated content.

### Evaluation Steps:
1. Thoroughly review both the table and the question, ensuring a full understanding of the information they convey. Identify and analyze key points, critical data, and important details within the table that is relevant to the question.
2. Carefully examine the proposed answer, focusing on its faithfulness. Check for factual correctness and verify whether the answer reflects and aligns with the information presented in the table.
3. Evaluate the answer's faithfulness using a strict 1 to 5 rating scale, with 1 being the lowest and 5 the highest.

Table:
{table}

Question:
{question}

Answer:
{gen_answer}
'''

COMP_PROMPT_TEMPLATE = '''G-Eval for Evaluating Comprehensiveness

### Task Introduction:
Given a complex question and a generated answer about a table, your task is to rate the answer's Comprehensiveness.

### Evaluation Criteria:
Comprehensiveness (1-5): A good answer should provide all the necessary information to address the question comprehensively. Additionally, it should avoid including details that, while consistent with the tabular data, are irrelevant to the given question.

### Evaluation Steps:
1. Carefully review the table and the question, ensuring you understand the full scope of the information provided. Identify all relevant points and details necessary to answer the question comprehensively.
2. Analyze the proposed answer to determine if it covers all the key aspects and addresses the question fully. Check whether the answer omits any important information or includes unnecessary details.
3. Evaluate the answer's comprehensiveness using a 1 to 5 rating scale, where 1 indicates the least comprehensive and 5 indicates the most.

Table:
{table}

Question:
{question}

Answer:
{gen_answer}
'''

MITIGATE_BOTH_PROMPT_TEMPLATE = """Mitigation Task: Improve Faithfulness and Completeness

### Role
You are a helpful assistant that revises long-form answers grounded in tabular data.

### Input
- A **table** of data
- A **question** about that table
- A model-generated **answer**
- Human-assigned **faithfulness score** and **completeness score** (each 1–4)

**Faithfulness score**: {faith_score}  
**Completeness score**: {comp_score}

### Task
The answer needs improvement in both **faithfulness** and **completeness**.  
Rewrite it so that it:
- Contains **only factual content** from the table (faithful)
- **Completely addresses** all relevant aspects of the question (comprehensive)
- Avoids any **unsupported** or **missing** information

### Output Format (STRICT)
Return **only** a JSON object in this exact shape—no extra text, comments, or markdown:

```json
{{
  "answer": "<your rewritten answer as a single text block>"
}}
```

### Table

{table}

### Question

{question}

### Original Answer

{model_answer}

Please output the corrected answer as JSON:
"""

MITIGATE_FAITH_ONLY_PROMPT_TEMPLATE = """Mitigation Task: Improve Faithfulness

### Role
You are a fact-checking assistant tasked with correcting factual inaccuracies.

### Input
* A **table** of data
* A **question** about that table
* A model-generated **answer**
* Human-assigned **faithfulness score** (1–4)

**Faithfulness score**: {faith_score}

### Task
The answer contains factual errors or ungrounded content.
Rewrite it to be:
* Fully factual
* Strictly grounded in the table
* Directly answering the question
* Without introducing unsupported content

### Output Format (STRICT)
Return only a JSON object in this exact shape—no extra text, comments, or markdown:

```json
{{
  "answer": "<your rewritten answer as a single text block>"
}}
```

### Table

{table}

### Question

{question}

### Original Answer

{model_answer}

Please output the corrected answer as JSON:
"""

MITIGATE_COMP_ONLY_PROMPT_TEMPLATE = """Mitigation Task: Improve Completeness

### Role
You are a helpful assistant tasked with revising incomplete answers.

### Input
* A **table** of data
* A **question** about that table
* A model-generated **answer**
* Human-assigned **completeness score** (1–4)

**Completeness score**: {comp_score}

### Task
The answer is missing key details or only partially addresses the question.
Rewrite it to:
* Include **all relevant facts** from the table needed to fully answer the question
* Be **comprehensive** without being verbose
* Avoid irrelevant details not related to the question

### Output Format (STRICT)
Return only a JSON object in this exact shape—no extra text, comments, or markdown:

```json
{{
  "answer": "<your rewritten answer as a single text block>"
}}
```

### Table

{table}

### Question

{question}

### Original Answer

{model_answer}

Please output the corrected answer as JSON:
""" 