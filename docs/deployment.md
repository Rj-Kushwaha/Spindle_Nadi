# Deploy to Streamlit Community Cloud

## Current deployment status

The app is ready to deploy, but this workspace currently has no Git repository,
GitHub remote, or configured GitHub CLI login. The project has not been
published online.

## Publish the source repository

1. Create a repository in your GitHub account. Choose **Public** if you want
   the source visible to everyone, or **Private** if you want source access
   restricted. Never include `.venv/`, credentials, `.streamlit/secrets.toml`,
   real machine data, or generated live/history logs.
2. Add that repository as the `origin` remote for this project and push the
   project files to its `main` branch. Make sure the generated synthetic
   training CSV is included; it is safe demo data and used by the app.
3. Verify that `requirements.txt`, `app.py`, and the project packages
   (`backend/`, `data/`, `frontend/`, `ml/`, and `simulator/`) are in the
   repository root.

## Create the Streamlit app

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with the
   GitHub account that can access the repository.
2. Create an app using the repository, the `main` branch, and `app.py` as the
   main file path.
3. Deploy and wait for dependency installation and app startup. Streamlit
   installs the packages listed in `requirements.txt`.
4. Open the generated app URL and test the fleet page, **Demo → warning**,
   **Demo → critical**, What-if, maintenance copilot, virtual maintenance,
   evidence timeline, and Reset.

## Runtime notes

- The app only needs packages in `requirements.txt`. The optional presentation
  generator has its own `requirements-presentation.txt` and is not imported by
  `app.py`.
- Live machine state belongs to each app session. Streamlit Community Cloud
  containers are not durable storage: live sensor, event, and maintenance CSV
  files can be lost when the app restarts and are not shared persistent history
  across deployments. For durable/shared records, connect a database or
  approved cloud storage before relying on it.
- This app contains no credentials and does not need secrets to run. Do not
  add real machine telemetry or secret keys to the public demo repository.
- The simulator, risk scores, maintenance copilot, virtual repairs, and RUL are
  demonstrations based on synthetic data. Do not connect this app to physical
  machine controls or use its predictions as safety decisions.
