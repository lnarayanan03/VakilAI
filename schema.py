# schema.py
from pydantic import BaseModel, Field, field_validator
from typing import List

class LegalAnswer(BaseModel):
    """
    Structured output schema for VakilAI.
    
    Block 5 concept: .with_structured_output(LegalAnswer) converts this
    Pydantic class → JSON Schema → sent as tool definition in API payload
    → LLM forced to return structured args → Pydantic validates → typed object
    
    This is NOT prompt engineering. This is a transport-level contract.
    """
    
    answer: str = Field(
        description="Plain language answer to the legal question. "
                    "Must be clear enough for a non-lawyer to understand."
    )
    
    applicable_law: str = Field(
        description="Name of the Act that applies. "
                    "Example: 'Indian Penal Code 1860' or 'RTI Act 2005'"
    )
    
    section_numbers: List[str] = Field(
        description="List of relevant section numbers cited. "
                    "Example: ['Section 302', 'Section 304']"
    )
    
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'. "
                    "High = answer clearly in context. "
                    "Low = partial information found."
    )
    
    found_in_context: bool = Field(
        description="True if answer was found in the retrieved legal documents. "
                    "False if LLM could not find it in context."
    )
    
    disclaimer: str = Field(
        default="This is for informational purposes only. "
                "For legal action, consult a qualified advocate.",
        description="Legal disclaimer — always included"
    )

    # Block 5 concept: field_validator with mode="before"
    # Runs BEFORE Pydantic's type check
    # Intercepts bad LLM output, cleans it, returns valid value
    # Prevents crashes when LLM returns unexpected values
    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
            if v in ["high", "medium", "low"]:
                return v
        return "medium"  # default if LLM returns something unexpected
    
    # Add this validator to LegalAnswer in schema.py

    @field_validator("found_in_context", mode="before")
    @classmethod
    def validate_found_in_context(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower().strip() == "true"  # "True" → True, "False" → False
        return True  # default to True

    @field_validator("section_numbers", mode="before")
    @classmethod
    def validate_sections(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            # Prefix bare numbers with "Section "
            result = []
            for item in v:
                if isinstance(item, str):
                    item = item.strip()
                    if item.isdigit():
                        item = f"Section {item}"
                result.append(item)
            return result
        return v


