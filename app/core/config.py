from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_NAME: str = "NextStep AI Interview Server"
    DEBUG: bool = True
    CORS_ORIGINS: str = (
        "http://localhost:5173,http://localhost:5174,http://localhost:8000,http://localhost:8002,"
        "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:8000,http://127.0.0.1:8002"
    )

    DATABASE_URL: str

    GROQ_API_KEY: str = ""
    GROQ_API_KEY_2: str = ""
    GROQ_API_KEY_3: str = ""
    GROQ_API_KEY_4: str = ""
    GROQ_API_KEY_5: str = ""
    GROQ_API_KEY_6: str = ""
    GROQ_API_KEY_7: str = ""
    GROQ_API_KEY_8: str = ""
    GROQ_API_KEY_9: str = ""
    GROQ_API_KEY_10: str = ""
    GROQ_API_KEY_11: str = ""
    GROQ_API_KEY_12: str = ""
    GROQ_API_KEY_13: str = ""
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""
    CLOUDFLARE_ACCOUNT_ID_2: str = ""
    CLOUDFLARE_API_TOKEN_2: str = ""
    CLOUDFLARE_ACCOUNT_ID_3: str = ""
    CLOUDFLARE_API_TOKEN_3: str = ""
    CLOUDFLARE_ACCOUNT_ID_4: str = ""
    CLOUDFLARE_API_TOKEN_4: str = ""
    CLOUDFLARE_ACCOUNT_ID_5: str = ""
    CLOUDFLARE_API_TOKEN_5: str = ""
    CLOUDFLARE_ACCOUNT_ID_6: str = ""
    CLOUDFLARE_API_TOKEN_6: str = ""
    CLOUDFLARE_ACCOUNT_ID_7: str = ""
    CLOUDFLARE_API_TOKEN_7: str = ""
    CLOUDFLARE_ACCOUNT_ID_8: str = ""
    CLOUDFLARE_API_TOKEN_8: str = ""
    CLOUDFLARE_ACCOUNT_ID_9: str = ""
    CLOUDFLARE_API_TOKEN_9: str = ""
    CLOUDFLARE_ACCOUNT_ID_10: str = ""
    CLOUDFLARE_API_TOKEN_10: str = ""
    CLOUDFLARE_ACCOUNT_ID_11: str = ""
    CLOUDFLARE_API_TOKEN_11: str = ""
    CLOUDFLARE_ACCOUNT_ID_12: str = ""
    CLOUDFLARE_API_TOKEN_12: str = ""
    CLOUDFLARE_ACCOUNT_ID_13: str = ""
    CLOUDFLARE_API_TOKEN_13: str = ""
    INCEPTION_API_KEY: str = ""
    INCEPTION_API_KEY_2: str = ""
    INCEPTION_API_KEY_3: str = ""
    INCEPTION_API_KEY_4: str = ""
    INCEPTION_API_KEY_5: str = ""
    INCEPTION_API_KEY_6: str = ""
    INCEPTION_API_KEY_7: str = ""
    INCEPTION_API_KEY_8: str = ""
    INCEPTION_API_KEY_9: str = ""
    INCEPTION_API_KEY_10: str = ""
    INCEPTION_API_KEY_11: str = ""
    INCEPTION_API_KEY_12: str = ""
    INCEPTION_API_KEY_13: str = ""
    OPENTYPHOON_API_KEY: str = ""
    OPENTYPHOON_API_KEY_2: str = ""
    OPENTYPHOON_API_KEY_3: str = ""
    OPENTYPHOON_API_KEY_4: str = ""
    OPENTYPHOON_API_KEY_5: str = ""
    OPENTYPHOON_API_KEY_6: str = ""
    OPENTYPHOON_API_KEY_7: str = ""
    OPENTYPHOON_API_KEY_8: str = ""
    OPENTYPHOON_API_KEY_9: str = ""
    OPENTYPHOON_API_KEY_10: str = ""
    OPENTYPHOON_API_KEY_11: str = ""
    OPENTYPHOON_API_KEY_12: str = ""
    OPENTYPHOON_API_KEY_13: str = ""
    OLLAMA_API_KEY: str = ""
    OLLAMA_API_KEY_2: str = ""
    OLLAMA_API_KEY_3: str = ""
    OLLAMA_API_KEY_4: str = ""
    OLLAMA_API_KEY_5: str = ""
    OLLAMA_API_KEY_6: str = ""
    OLLAMA_API_KEY_7: str = ""
    OLLAMA_API_KEY_8: str = ""
    OLLAMA_API_KEY_9: str = ""
    OLLAMA_API_KEY_10: str = ""
    OLLAMA_API_KEY_11: str = ""
    OLLAMA_API_KEY_12: str = ""
    OLLAMA_API_KEY_13: str = ""
    NVIDIA_API_KEY: str = ""
    NVIDIA_API_KEY_2: str = ""
    NVIDIA_API_KEY_3: str = ""
    NVIDIA_API_KEY_4: str = ""
    NVIDIA_API_KEY_5: str = ""
    NVIDIA_API_KEY_6: str = ""
    NVIDIA_API_KEY_7: str = ""
    NVIDIA_API_KEY_8: str = ""
    NVIDIA_API_KEY_9: str = ""
    NVIDIA_API_KEY_10: str = ""
    NVIDIA_API_KEY_11: str = ""
    NVIDIA_API_KEY_12: str = ""
    NVIDIA_API_KEY_13: str = ""
    GEMINI_API_KEY: str = ""

    GROQ_MODEL: str = "openai/gpt-oss-120b"
    CLOUDFLARE_MODEL: str = "@cf/qwen/qwen3-30b-a3b-fp8"
    INCEPTION_MODEL: str = "mercury-2.5"
    OPENTYPHOON_MODEL: str = "typhoon-v2.5-30b-a3b-instruct"
    OLLAMA_CLOUD_MODEL: str = "gemma4:31b"
    NVIDIA_NEMOTRON_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    DATASET_PROVIDER_TOKEN_BUDGET: int = 0
    CV_AI_ENRICHMENT_ENABLED: bool = True
    INTERVIEW_MULTI_PROVIDER_ENABLED: bool = True
    INTERVIEW_QUESTION_CANDIDATE_COUNT: int = 4
    TTS_PREWARM_ENABLED: bool = True
    TTS_PREWARM_URL: str = "http://localhost:8000/api/v1/tts/prewarm"
    TTS_PREWARM_NOTIFY_TIMEOUT_SECONDS: float = 3.0
    INTERVIEW_QUESTION_RANKER_ARTIFACT: str = (
        "artifacts/weak_supervision/v1_1/question_pairwise_ranker_v2.joblib"
    )
    INTERVIEW_ANSWER_SCORER_ARTIFACT: str = (
        "artifacts/weak_supervision/v1_1/candidate_answer_scorer.joblib"
    )
    RESEARCH_PSEUDONYM_SALT: str = ""

    JWT_ACCESS_SECRET: str
    JWT_ACCESS_EXPIRES_IN: str

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", BASE_DIR / ".env.production"),
        extra="ignore",
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_value(cls, value):
        if isinstance(value, bool):
            return value

        normalized = str(value or "").strip().lower()
        if normalized in {"1", "true", "yes", "on", "debug", "development", "dev"}:
            return True
        if normalized in {"0", "false", "no", "off", "release", "production", "prod"}:
            return False
        return value

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def sanitize_database_url(cls, value):
        raw = str(value or "").strip()
        if not raw:
            return raw

        # Some external tools append pgbouncer=true, which psycopg2 rejects.
        if "?" not in raw or "pgbouncer" not in raw.lower():
            return raw

        base, query = raw.split("?", 1)
        kept_parts: list[str] = []
        for part in query.split("&"):
            key = part.split("=", 1)[0].strip().lower()
            if key == "pgbouncer":
                continue
            if part.strip():
                kept_parts.append(part.strip())

        return f"{base}?{'&'.join(kept_parts)}" if kept_parts else base


settings = Settings()
