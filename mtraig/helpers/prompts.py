CLAIM_DECOMPOSITION_PROMPT = '''You are a helpful assistant tasked with decomposing a given insight according to a table schema. Your goal is to break the given insight down into atomic-level claims.

Task Description:

- You will be provided with a table schema and an insight.
- Your goal is to decompose the given insight into atomic-level claims **only for the parts that can be answered based on the table schema**, preserving the original wording wherever possible.
- The table schema indicates how the data in the table is structured, and you have an insight that references the table.

Instructions:

1. Read the **Table Schema** and the **Insight** carefully.
2. Identify which parts of the insight can be supported or answered using the table schema.
3. Break those parts into atomic-level claims that reflect the insight relevant to the provided schema, preserving the original wording as much as possible.
4. Present your output as a numbered list of atomic-level claims.
5. Do not mention table name, column names, or provide any additional text or explanation.

Table Schema:
{schema}

Insight:
{insight}

Output:'''

CLAIM_VERIFICATION_PROMPT = '''You will be given a table and a claim.

Your task is to verify whether the claim is faithful to the data in the given table.

Please read and follow these instructions carefully.

Evaluation Criteria

Faithfulness (0 or 1)
- 0: Claim contains contradictions, inaccuracies, or information that is not explicitly supported by the table.
- 1: Claim adheres to the table data and contains no contradictions or unsupported elements.

Evaluation Steps:

1. Examine the table: Identify the essential data relevant to the claim.
2. Compare the claim: Check for contradictions, inaccuracies, or any details not supported by the table.
3. Decide Faithfulness: If the claim fully aligns with the table data, assign 1; otherwise, assign 0.

Table:
{table}

Claim:
{claim}

Evaluation Form (scores ONLY):
- Faithfulness:'''

# ------------------------------------------------------------------------------
# ② COARSE‑LEVEL MITIGATION PROMPT
# ------------------------------------------------------------------------------

MTRAIG_MITIGATION_PROMPT_TEMPLATE = """MT-RAIG Mitigation Task

### Role
You are a fact-correcting assistant that corrects answers to ensure they are perfectly faithful to the table data.

### Input
- A **table** of data
- A **question** about that table
- The model-generated **answer**
- A list of **false claims** that were identified in the answer

### Task
Rewrite the answer to address the false claims while ensuring the response is:
1. Completely factual and grounded only in the table
2. Directly answers the question
3. Does not include any information that cannot be verified in the table

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

### False Claims to Address

{false_claims}

Please output the corrected answer as JSON:
"""


