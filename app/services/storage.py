import os
import shutil
from fastapi import UploadFile
from app.config import settings

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None

class StorageService:
    def __init__(self):
        self.supabase: Client | None = None
        if settings.SUPABASE_URL and settings.SUPABASE_KEY and create_client:
            try:
                self.supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            except Exception as e:
                print(f"Warning: Supabase init failed: {e}")

    async def save_file(self, file: UploadFile) -> dict:
        """
        Saves file locally and optionally uploads to Supabase.
        Returns a dict with file path and status.
        """
        # 1. Save locally
        file_location = os.path.join(settings.UPLOAD_DIR, file.filename)
        
        # Reset file cursor just in case
        await file.seek(0)
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = {
            "filename": file.filename,
            "local_path": file_location,
            "supabase_upload": "skipped"
        }

        # 2. Upload to Supabase (if configured)
        if self.supabase:
            try:
                with open(file_location, "rb") as f:
                    file_bytes = f.read()
                    # Determine bucket? Let's assume 'uploads' bucket exists or just print log
                    # self.supabase.storage.from_("uploads").upload(file.filename, file_bytes)
                    result["supabase_upload"] = "configured_but_bucket_pending" 
            except Exception as e:
                result["supabase_upload"] = f"failed: {str(e)}"
        
        return result

storage_service = StorageService()
