"""
Pydantic schemas for G-Eval detection structured output.
"""

from pydantic import BaseModel

class FaithfulnessScore(BaseModel):
    faithfulness: int  # integer 1‑5

class CompletenessScore(BaseModel):
    completeness: int  # integer 1‑5 

class AnswerRewrite(BaseModel):
    answer: str 