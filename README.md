# BlueRayScan Backend

FastAPI backend for BlueRayScan deepfake monitoring system.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in `backend/` (optional for local run, required for Supabase):
   ```
   SUPABASE_URL=your_url
   SUPABASE_KEY=your_key
   ```

3. **Run Server**
   ```bash
   # From backend directory
   uvicorn app.main:app --reload
   ```

## Endpoints

- `GET /` : Health check
- `POST /api/scan` : Upload file for analysis
- `GET /docs` : Swagger UI API documentation
