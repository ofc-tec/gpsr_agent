from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict

ActionName = Literal[
    "navigate_to", "detect_object", "detect_person", "grasp",
    "release", "follow_person", "say", "listen","count_objects",
]

class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: ActionName
    args: dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str
    steps: list[Step] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
