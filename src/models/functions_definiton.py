from pydantic import BaseModel
from typing import Dict


class Parameter(BaseModel):
    type: str


class ReturnType(BaseModel):
    type: str


class FunctionDefintion(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: ReturnType