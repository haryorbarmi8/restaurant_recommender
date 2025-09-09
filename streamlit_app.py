# streamlit_app.py
# -------------------------------------------------------------
# Restaurant Recommender with PostgreSQL backend (secure connection)
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import sqlalchemy
import urllib.parse

st.set_page_config(page_title="🍽️ Restaurant Recommender", page_icon="🍽️", layout="wide")

# ---------------------------
# Database Connection (from secrets)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_engine():
    """Create a SQLAlchemy engine for PostgreSQL using Streamlit secrets."""
    user = st.secrets["postgres"]["user"]
    password = urllib.parse.quote_plus(st.secrets["postgres"]["password"])
    host = st.secrets["postgres"]["host"]
    port = st.secrets["postgres"].get("port", 5432)
    dbname = st.secrets["postgres"]["dbname"]

    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return sqlalchemy.create_engine(db_url)


@st.cache_data(show_spinner=False)
def load_data(engine, min_rating: float):
    query = """
        SELECT "Restaurant ID", "Restaurant Name", "Cuisines", 
               "Price range", "Aggregate rating", "Votes"
        FROM restaurants
        WHERE "Aggregate rating" >= %(min_rating)s
    """
    df = pd.read_sql(query, engine, params={"min_rating": min_rating})

    if df.empty:
        return df

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
def preprocess(df: pd.DataFrame):
    tokens = df['Cuisines'].str.split(r',\s*', regex=True)
    df_exp = df.assign(Cuisines=tokens).explode('Cuisines')
    df_exp['Cuisines'] = df_exp['Cuisines'].astype(str).str.strip()
    df_exp = df_exp[df_exp['Cuisines'] != '']

    restoXcuisines = pd.crosstab(df_exp['Restaurant Name'], df_exp['Cuisines'])
    restoXcuisines = (restoXcuisines > 0).astype(np.uint8)

    meta = (
        df.sort_values(['Restaurant Name', 'Aggregate rating'], ascending=[True, False])
          .drop_duplicates('Restaurant Name', keep='first')
          .set_index('Restaurant Name')
          [['Aggregate rating', 'Price range', 'Votes', 'Restaurant ID']]
    )

    cuisine_lists = (
        df_exp.groupby('Restaurant Name')['Cuisines']
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
st.title("🍽️ Restaurant Recommender (PostgreSQL)")
st.caption("Find restaurants that match your taste, always updated from the database.")

with st.sidebar:
    st.header("Filters")
    min_rating = st.slider("Minimum aggregate rating", 0.0, 5.0, 4.0, 0.1)
    min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.7, 0.05)
    k = st.slider("# of recommendations", 1, 20, 5)

try:
    engine = get_engine()
    df = load_data(engine, min_rating)

    if df.empty:
        st.warning("No restaurants meet the rating filter. Add more data or lower the filter.")
        st.stop()

    crosstab, meta, cuisine_lists = preprocess(df)
    sim_df = jaccard_sim(crosstab)

    restaurant = st.selectbox("Choose a restaurant:", sorted(sim_df.index))

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

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Tip: Make sure your PostgreSQL database has a 'restaurants' table with columns: Restaurant ID, Restaurant Name, Cuisines, Price range, Aggregate rating, Votes.")
