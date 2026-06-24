# 🎬 CineIQ — Smart Movie Recommender

A content-based movie recommendation engine built with **TF-IDF + Cosine Similarity**, powered by the **TMDB API** for real-time movie data.

## Features

- 🎯 **ML Recommendations** — TF-IDF vectorization with bigrams, industry-aware penalty/bonus scoring
- 🎬 **TMDB Integration** — Auto-fetches posters, ratings, cast, director, plot, and keywords
- ➕ **Self-improving DB** — Add any movie via search; tags are auto-built from TMDB data
- 🔍 **Genre Filter** — Filter your library by genre/industry before picking
- 🌙 **Dark Mode** — Toggle via sidebar
- 🔐 **Admin Panel** — Password-protected movie management (add/remove)

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| ML Model | scikit-learn (TF-IDF, Cosine Similarity) |
| Data | TMDB API v3 |
| Storage | JSON flat file |
| Language | Python 3.10+ |

## Setup (Local)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cineiq.git
cd cineiq

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your secrets
# Create .streamlit/secrets.toml with:
# TMDB_API_KEY = "your_key_here"
# ADMIN_PASSWORD = "your_password_here"

# 4. Run
streamlit run app.py
```

> Get a free TMDB API key at https://www.themoviedb.org/settings/api

## Deployment (Streamlit Cloud)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo
3. Add secrets in the Streamlit Cloud dashboard under **Settings → Secrets**:
```toml
TMDB_API_KEY = "your_key_here"
ADMIN_PASSWORD = "your_password_here"
```

## How the Model Works

```
User selects a movie
        ↓
TF-IDF vectorizes all genre/tag strings (bigrams)
        ↓
Cosine similarity computed across all movie pairs
        ↓
Industry penalty (−0.25) if cross-industry match
Genre overlap bonus (+0.08) if 2+ genres match
        ↓
Dynamic threshold filters weak matches (≥65% of top score)
        ↓
Top N recommendations returned
```

## Project Structure

```
movie_recommender/
├── app.py              # Main Streamlit app
├── movies.json         # Movie database (auto-updated)
├── requirements.txt    # Python dependencies
├── README.md
└── .streamlit/
    ├── config.toml     # UI theme
    └── secrets.toml    # API keys (NOT committed to git)
```

---
Built by **Ayush** · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
