from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Platform(StrEnum):
    android = "android"
    ios = "ios"
    react_native = "react_native"
    flutter = "flutter"


class Settings(BaseModel):
    target_path: Path = Field(..., description="Path to the target mobile app source tree.")
    platform: Platform = Field(..., description="Target mobile platform.")
    model_name: str = Field(
        "BAAI/bge-small-en-v1.5",
        description="Sentence-transformers model for local embeddings.",
    )
    output_dir: str = Field(
        ".klit-flow",
        description="Output directory name, relative to target_path.",
    )
    summaries: bool = Field(
        False,
        description="Enable optional NL summaries via a local Ollama model.",
    )
