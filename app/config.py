import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "BlueRayScan Backend"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Storage
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "backend", "uploads")
    
    # Supabase (Optional for now, but ready)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
