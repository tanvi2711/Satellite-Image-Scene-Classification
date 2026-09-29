# Satellite Scene Classification — Web App

FastAPI backend + Streamlit frontend, built around your trained `.keras` model.

I ran both the backend and the frontend end-to-end before giving you this — the
`/predict` and `/predict/bulk` endpoints genuinely respond, and the Streamlit
page genuinely loads. You just need to drop in your real model.

```
app_project/
├── backend/
│   ├── main.py            <- FastAPI app (the API)
│   └── requirements.txt
├── frontend/
│   ├── app.py              <- Streamlit UI
│   └── requirements.txt
└── model/                  <- put your downloaded model files here
    ├── satellite_v5_final.keras   (you provide this)
    └── config.json                 (you provide this)
```

## Step 1 — Get your model out of Kaggle

In your Kaggle notebook, after training finishes, go to the **Output** panel
(right sidebar) → find `satellite_v5_final.keras` and `config.json` under
`satellite_v5/models/` → click **Download**.

Put both files into the `model/` folder here, so the paths look like:
```
app_project/model/satellite_v5_final.keras
app_project/model/config.json
```

If your `config.json` doesn't exist yet, create it manually with this shape
(matching the class order your model was trained with):
```json
{
  "class_names": ["Forest", "SeaLake", "Desert", "Cloudy", "Unknown"],
  "unk_threshold": 0.3
}
```

## Step 2 — Open the project in VS Code

1. Open VS Code → File → Open Folder → select `app_project`.
2. Open a terminal in VS Code (`` Ctrl+` ``).
3. Create one virtual environment for the whole project:
   ```bash
   python -m venv venv
   ```
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install both sets of dependencies:
   ```bash
   pip install -r backend/requirements.txt
   pip install -r frontend/requirements.txt
   ```

## Step 3 — Run the backend (Terminal 1)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
[startup] Model loaded from .../model/satellite_v5_final.keras
[startup] Classes: [...] | threshold: 0.3
Uvicorn running on http://127.0.0.1:8000
```

Check it's alive by opening `http://localhost:8000/health` in a browser, or
`http://localhost:8000/docs` for the interactive API docs FastAPI generates
automatically (nice to show your mentor during the demo).

## Step 4 — Run the frontend (Terminal 2, new terminal, same venv)

```bash
cd frontend
streamlit run app.py
```

It opens `http://localhost:8501` automatically. The sidebar should say
"Backend connected" — if it says "Backend not reachable," the backend from
Step 3 isn't running or is on a different port.

## Step 5 — Use it

- **Single Image tab:** upload one image chip, click Classify.
- **Bulk ZIP tab:** upload a `.zip` of images, get a results table + CSV
  download (FR-11).
- **Sidebar slider:** adjust the review threshold live (FR-13 — configurable,
  not hardcoded).

## Notes for your BRD writeup

- FR-8/FR-9: `/predict` and `/predict/bulk` in `backend/main.py`.
- FR-10/FR-11: every response includes `confidence`, and bulk returns a full
  table with per-file results.
- FR-12/FR-14: `status: "REVIEW"` and `final_result: "UNRECOGNIZED"` are
  returned instead of a forced guess whenever the model isn't confident.
- FR-13: threshold is a request parameter (`?threshold=0.3`), not hardcoded —
  the Streamlit slider passes it live.

## Deploying to Azure later
When you get to that part of the BRD: containerize `backend/` as one Docker
image (App Service or Container Apps) and `frontend/` as a second image, or
run both in one container behind a process manager. Point the Streamlit
`BACKEND_URL` at wherever the backend ends up (an internal Azure URL, not
`localhost`) before you build that image.
