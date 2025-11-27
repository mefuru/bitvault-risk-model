# Deployment Guide: Streamlit Community Cloud

This guide walks you through deploying the BitVault Risk Model to Streamlit Community Cloud.

---

## Prerequisites

1. **GitHub account** - Your code needs to be in a GitHub repository
2. **FRED API key** - Free from https://fred.stlouisfed.org/docs/api/api_key.html
3. **Streamlit account** - Free at https://share.streamlit.io

---

## Step 1: Prepare Your Repository

### 1.1 Ensure these files exist in your repo root:

```
bitvault-risk-model/
├── src/                      # All source code
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── requirements.txt          # Python dependencies
├── .gitignore               # Excludes data/, logs/, .env
└── README.md
```

### 1.2 Verify `.gitignore` excludes sensitive files:

```gitignore
# These should NOT be committed
data/*.db
logs/*.log
.env
.streamlit/secrets.toml
```

### 1.3 Push to GitHub:

```bash
cd bitvault-risk-model

# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for Streamlit Cloud deployment"

# Create GitHub repo and push (using GitHub CLI)
gh repo create bitvault-risk-model --private --push

# Or manually:
# 1. Create repo at github.com
# 2. git remote add origin https://github.com/YOUR_USERNAME/bitvault-risk-model.git
# 3. git push -u origin main
```

---

## Step 2: Deploy to Streamlit Cloud

### 2.1 Go to [share.streamlit.io](https://share.streamlit.io)

- Sign in with your GitHub account
- Authorize Streamlit to access your repositories

### 2.2 Click "New app"

Fill in:
- **Repository**: `YOUR_USERNAME/bitvault-risk-model`
- **Branch**: `main`
- **Main file path**: `src/dashboard/app.py`

### 2.3 Click "Deploy"

The first deployment takes 3-5 minutes as it:
1. Clones your repo
2. Installs dependencies from `requirements.txt`
3. Starts the Streamlit app

---

## Step 3: Configure Secrets

Your app needs a FRED API key for macro data. Add it as a secret:

### 3.1 In Streamlit Cloud dashboard:
- Click on your app
- Click "Settings" (gear icon)
- Go to "Secrets" section

### 3.2 Add your secrets:

```toml
FRED_API_KEY = "your_actual_fred_api_key_here"
```

### 3.3 Click "Save"

The app will restart automatically.

---

## Step 4: First Run

On first load, the app will:

1. **Initialize the database** - Creates SQLite file
2. **Backfill 3 years of data** - Takes ~30 seconds
3. **Display the dashboard** - Ready to use

You'll see a spinner: "Initializing data (first run only, ~30 seconds)..."

After this, subsequent loads are instant (cached).

---

## Understanding Streamlit Cloud Behavior

### Data Persistence

⚠️ **Important**: Streamlit Cloud's filesystem is **ephemeral**. This means:

- The SQLite database resets on every deploy or app restart
- The app automatically re-fetches data when this happens
- First load after a reset takes ~30 seconds

This is fine for a dashboard that uses live data. If you need persistent storage, consider:
- Railway ($5/month)
- Render with persistent disk
- Self-hosted on a VPS

### App Sleeping

On the free tier:
- Apps sleep after ~7 days of inactivity
- First visit after sleep has a ~30 second cold start
- The app then re-initializes data

### Resource Limits

Free tier limits:
- 1 GB RAM
- 1 CPU
- Apps are public (URL can be shared)

Your app uses ~200-400 MB RAM, well within limits.

---

## Updating Your App

To update the deployed app:

```bash
# Make changes locally
git add .
git commit -m "Description of changes"
git push origin main
```

Streamlit Cloud automatically redeploys when you push to the main branch.

---

## Troubleshooting

### "ModuleNotFoundError"

**Cause**: Missing dependency in `requirements.txt`

**Fix**: Add the missing package to `requirements.txt` and push

### "No data available"

**Cause**: Data initialization failed

**Fix**: 
1. Check Streamlit Cloud logs (Settings → Logs)
2. Verify FRED_API_KEY is set in secrets
3. Try rebooting the app (Settings → Reboot app)

### App crashes on load

**Cause**: Usually a code error

**Fix**:
1. Check logs in Streamlit Cloud dashboard
2. Test locally first: `PYTHONPATH=. streamlit run src/dashboard/app.py`
3. Fix errors and push

### "API key not configured"

**Cause**: Secret not set correctly

**Fix**:
1. Go to Settings → Secrets
2. Ensure format is exactly: `FRED_API_KEY = "your_key"`
3. No spaces around `=`, quotes around value
4. Save and wait for app restart

---

## Optional: Password Protection

To restrict access to your app:

### 3.1 Add to `.streamlit/secrets.toml` in Streamlit Cloud:

```toml
FRED_API_KEY = "your_key"

[passwords]
admin = "your_secure_password"
```

### 3.2 Add authentication code to `app.py`:

```python
import streamlit as st

def check_password():
    """Returns True if password is correct."""
    def password_entered():
        if st.session_state["password"] == st.secrets["passwords"]["admin"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# Rest of your app...
```

---

## Summary

| Step | Action | Time |
|------|--------|------|
| 1 | Push code to GitHub | 2 min |
| 2 | Deploy on Streamlit Cloud | 5 min |
| 3 | Add FRED_API_KEY secret | 1 min |
| 4 | Wait for first data load | 30 sec |

**Total: ~10 minutes to deploy**

Your app URL will be: `https://YOUR_APP_NAME.streamlit.app`

---

## Alternative: Railway Deployment

If you need persistent data storage, Railway is recommended:

1. Go to [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub repo"
3. Select your repo
4. Add environment variable: `FRED_API_KEY`
5. Railway auto-detects Streamlit and deploys

Cost: ~$5/month for always-on with persistent storage.
