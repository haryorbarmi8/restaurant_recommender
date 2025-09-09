# streamlit_app.py
# -------------------------------------------------------------
# Restaurant Recommender (pure NumPy/Pandas — no SciPy needed)
# Deployable on Streamlit Community Cloud
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🍽️ Restaurant Recommender", page_icon="🍽️", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes | None):
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        try:
            df = pd.read_csv("Cognify1.csv")
        except Exception as e:
            raise FileNotFoundError(
                "No data found. Upload a CSV or place 'Cognify1.csv' in the app folder.") from e

    required_cols = [
        'Restaurant ID', 'Restaurant Name', 'Cuisines',
        'Price range', 'Aggregate rating', 'Votes'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV must contain columns: {required_cols}. Missing: {missing}")

    df = df[required_cols].copy()
    df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')
    df['Price range'] = pd.to_numeric(df['Price range'], errors='coerce')
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

    df['Cuisines'] = df['Cuisines'].astype(str)
    df['Cuisines'] = df['Cuisines'].str.replace(r"[\[\]\{\}\(\)\"']", '', regex=True)
    df['Cuisines'] = df['Cuisines'].str.replace(';', ',')
    df['Cuisines'] = df['Cuisines'].str.replace(' ,', ',')
    df['Cuisines'] = df['Cuisines'].str.replace(r',\s*', ', ', regex=True)

    df = df.dropna(subset=['Restaurant Name', 'Cuisines', 'Aggregate rating'])
    return df


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, min_rating: float):
    dff = df[df['Aggregate rating'] >= float(min_rating)].copy()
    tokens = dff['Cuisines'].str.split(r',\s*', regex=True)
    df_exp = dff.assign(Cuisines=tokens).explode('Cuisines')
    df_exp['Cuisines'] = df_exp['Cuisines'].astype(str).str.strip()
    df_exp = df_exp[df_exp['Cuisines'] != '']

    restoXcuisines = pd.crosstab(df_exp['Restaurant Name'], df_exp['Cuisines'])
    restoXcuisines = (restoXcuisines > 0).astype(np.uint8)

    meta = (
        dff.sort_values(['Restaurant Name', 'Aggregate rating'], ascending=[True, False])
           .drop_duplicates('Restaurant Name', keep='first')
           .set_index('Restaurant Name')
           [['Aggregate rating', 'Price range', 'Votes', 'Restaurant ID']]
    )

    cuisine_lists = (df_exp.groupby('Restaurant Name')['Cuisines']
                          .agg(lambda s: sorted(set(x for x in s if isinstance(x, str) and x.strip())))
                    )

    return restoXcuisines, meta, cuisine_lists


@st.cache_data(show_spinner=False)
def jaccard_sim(crosstab: pd.DataFrame) -> pd.DataFrame:
    if crosstab.empty:
        return pd.DataFrame()

    X = crosstab.values.astype(np.uint8)
    intersections = X @ X.T
    row_sums = X.sum(axis=1)
    unions = row_sums[:, None] + row_sums[None, :] - intersections
    unions = np.where(unions == 0, 1, unions)

    sim = intersections / unions
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=crosstab.index, columns=crosstab.index)


# ---------------------------
# UI
# ---------------------------
st.title("🍽️ Restaurant Recommender")
st.caption("Select a restaurant to get similar picks based on shared cuisines and compare ratings.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload your restaurant CSV", type=["csv"])
    file_bytes = uploaded.getvalue() if uploaded else None

    st.header("Filters")
    min_rating = st.slider("Minimum aggregate rating", 0.0, 5.0, 4.0, 0.1)
    min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.7, 0.05)
    k = st.slider("# of recommendations", 1, 20, 5)

try:
    df = load_data(file_bytes)
    crosstab, meta, cuisine_lists = preprocess(df, min_rating)
    sim_df = jaccard_sim(crosstab)

    if sim_df.empty:
        st.warning("No restaurants left after filtering. Lower the rating threshold or upload a different dataset.")
        st.stop()

    restaurant = st.selectbox("Choose a restaurant:", sorted(sim_df.index))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Restaurants", len(sim_df.index))
    with m2:
        st.metric("Unique cuisines", crosstab.shape[1])
    with m3:
        st.metric("Min rating filter", f"{min_rating:.1f}")

    if restaurant:
        sims = sim_df.loc[restaurant].sort_values(ascending=False).drop(index=restaurant)
        sims = sims[sims >= min_sim].head(k)

        rec = pd.DataFrame({
            'Restaurant Name': sims.index,
            'Similarity': np.round(sims.values.astype(float), 3)
        })
        rec = rec.join(meta, on='Restaurant Name')

        base_set = set(cuisine_lists.get(restaurant, []))
        def shared(name: str) -> str:
            return ', '.join(sorted(base_set.intersection(set(cuisine_lists.get(name, [])))))

        rec['Shared Cuisines'] = rec['Restaurant Name'].apply(shared)
        rec = rec.sort_values(['Similarity', 'Aggregate rating'], ascending=[False, False]).reset_index(drop=True)

        st.subheader("Recommended Restaurants 🍴")
        st.dataframe(rec, use_container_width=True)

        csv = rec.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download recommendations (CSV)",
            data=csv,
            file_name=f"recs_{restaurant.replace(' ', '_')}.csv",
            mime="text/csv"
        )


    with st.expander("Preview first 10 restaurants"):
        st.dataframe(meta.reset_index().head(10), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Tip: Ensure the CSV has columns: Restaurant ID, Restaurant Name, Cuisines, Price range, Aggregate rating, Votes.")
