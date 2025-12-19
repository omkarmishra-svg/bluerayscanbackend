# Deploying BlueRayScan Backend

**Scenario**: You have pushed the contents of the `backend/` folder to a **NEW, SEPARATE GitHub Repository**.

## Option 1: Render (Recommended)

1.  Connect your **Backend Repository**.
2.  **Settings**:
    *   **Root Directory**: Leave blank (Use default).
    *   **Runtime**: Python 3.
    *   **Build Command**: `pip install -r requirements.txt`.
    *   **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3.  **Environment Variables**:
    *   Add `SUPABASE_URL` and `SUPABASE_KEY`.
4.  Click **Deploy**.

## Option 2: Railway`

1.  **+ New Project** -> **Deploy from GitHub repo**.
2.  Select your **Backend Repository**.
3.  Railway will automatically detect `Procfile` and `requirements.txt` at the root.
4.  Add variables (`SUPABASE_URL`, etc).

## Option 3: Heroku

1.  `heroku create`.
2.  `git push heroku main`.
3.  (Since `Procfile` is at the root, it just works).

## Important: CORS
Since Frontend and Backend are on different domains (e.g. `vercel.app` vs `onrender.com`), ensure `main.py` allows the frontend URL.
*   Currently `main.py` is set to `allow_origins=["*"]`, which is perfect for hackathons.

## Note on "Mock Mode"
If the cloud server lacks specific GPU libraries, the backend will automatically use the **Safe Mode (Mock)** fallback.

