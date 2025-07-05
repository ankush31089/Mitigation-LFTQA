from typing import List
from pydantic import BaseModel, Field

class ClaimDecompositionResult(BaseModel):
    claims: List[str] = Field(description="A list of atomic-level claims extracted from the insight.")

class ClaimVerificationResult(BaseModel):
    faithfulness: int = Field(description="0 if the claim is unfaithful, 1 if it is faithful")

class AnswerRewrite(BaseModel):
    answer: str 