from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


# Load .env (optional). If you run locally, you can also hardcode values.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=str(ENV_PATH), override=False)

class Settings(BaseSettings):
    # Paths
    project_root: Path = Field(default=PROJECT_ROOT)
    index_path: Path = Field(default=PROJECT_ROOT / "output" / "pubmed_ivfpq.faiss")
    ondisk_dir: Path = Field(default=PROJECT_ROOT / "output" / "ondisk_lists")
    data_dir: Path = Field(default=PROJECT_ROOT / "data" / "MedCPT")  # contains pubmed/, pmids/, embeds/...
    cases_dir: Path = Field(default=PROJECT_ROOT / "Cases")

    # Retrieval
    nprobe: int = 32
    topk: int = 20
    show_metadata: bool = True
    prefer_abstract_topk: int = 5
    model_name: str = "ncbi/MedCPT-Query-Encoder"
    use_ondisk: bool = False  # must match how you built the index

    # Generation / LLM
    anthropic_model: str = "claude-3-haiku-20240307"
    anthropic_api_key: str | None = None
    anthropic_max_tokens: int = 1200
    temperature: float = 0.2

    class Config:
        model_config = SettingsConfigDict(
        env_prefix="",        # read ANTHROPIC_API_KEY etc. directly
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()
settings.cases_dir.mkdir(parents=True, exist_ok=True)