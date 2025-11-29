"""
Social Media Analytics Dashboard (Streamlit)

Save file as: social_media_dashboard.py

Required columns if uploading CSV:
- platform (e.g., Instagram, TikTok)
- date (YYYY-MM-DD)
- post_id (unique id for post)
- post_type (e.g., photo, video, reel)
- impressions (int)
- likes (int)
- comments (int)
- shares (int)
- followers (optional; follower count at that date)
- clicks (optional; used for CTR if present)

If followers not provided per-post, follower-growth metrics will be estimated from available rows per platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Social Media Analytics", layout="wide")
st.title("📱 Social Media Analytics Dashboard")

# ---------- Sidebar ----------
st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use sample data", value=True)
uploaded_file = st.sidebar.file_uploader("Upload social CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Settings")
resample_freq = st.sidebar.selectbox("Aggregate frequency", options=["D", "W", "M"], index=2,
                                     help="D = daily, W = weekly, M = monthly")
top_n = st.sidebar.slider("Top N posts to show", 3, 20, 5)

st.sidebar.markdown("---")
st.sidebar.header("Export")
export_agg = st.sidebar.checkbox("Include aggregated metrics CSV", value=True)


# ---------- Sample data generator ----------
def make_sample_social_data():
    rng = np.random.default_rng(42)
    platforms = ["Instagram", "TikTok", "Facebook", "YouTube"]
    post_types = ["photo", "video", "reel"]
    start = pd.to_datetime("2025-01-01")
    rows = []
    post_id = 1
    for day in pd.date_range(start, periods=90, freq="D"):
        for platform in platforms:
            # simulate number of posts per day per platform (0-3)
            n_posts = rng.integers(0, 3)
            for _ in range(n_posts):
                impressions = int(max(50, rng.poisson(2000)))
                likes = int(impressions * rng.uniform(0.02, 0.12))
                comments = int(impressions * rng.uniform(0.001, 0.01))
                shares = int(impressions * rng.uniform(0.0005, 0.005))
                clicks = int(impressions * rng.uniform(0.005, 0.03))
                followers = int(1000 + (day - start).days * rng.uniform(1, 5) + rng.integers(-10, 10))
                post_type = rng.choice(post_types, p=[0.4, 0.4, 0.2])
                rows.append({
                    "platform": platform,
                    "date": day,
                    "post_id": f"P{post_id}",
                    "post_type": post_type,
                    "impressions": impressions,
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "clicks": clicks,
                    "followers": followers,
                    "caption": f"Sample post {post_id} on {platform}"
                })
                post_id += 1
    df = pd.DataFrame(rows)
    return df


# ---------- Load data ----------
if use_sample:
    df = make_sample_social_data()
else:
    df = None
    if uploaded_file:
        try:
            raw = uploaded_file.read().decode("utf-8")
            df = pd.read_csv(StringIO(raw))
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

if df is None:
    st.info("Upload a CSV or enable sample data on the left.")
    st.stop()

# ---------- Normalize & clean ----------
# Ensure date column is datetime
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
else:
    st.error("CSV must include a `date` column.")
    st.stop()

# Fill missing numeric metrics with zeros
numeric_cols = ["impressions", "likes", "comments", "shares", "clicks", "followers"]
for c in numeric_cols:
    if c not in df.columns:
        df[c] = np.nan
df[numeric_cols] = df[numeric_cols].fillna(0).astype(float)

# Derived metrics per post
df["engagements"] = df["likes"] + df["comments"] + df["shares"]
# engagement rate = engagements / impressions (guard divide-by-zero)
df["engagement_rate"] = np.where(df["impressions"] > 0, df["engagements"] / df["impressions"], 0)
df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], np.nan)

# Basic filters
platforms = df["platform"].unique().tolist()
selected_platforms = st.multiselect("Platforms", options=platforms, default=platforms)
post_types = df["post_type"].unique().tolist()
selected_post_types = st.multiselect("Post types", options=post_types, default=post_types)
date_min, date_max = st.date_input("Date range", [df["date"].min(), df["date"].max()])

mask = (df["platform"].isin(selected_platforms)) & (df["post_type"].isin(selected_post_types)) & \
       (df["date"] >= pd.to_datetime(date_min)) & (df["date"] <= pd.to_datetime(date_max))
dff = df.loc[mask].copy()

if dff.empty:
    st.warning("No data for selected filters. Adjust filters or date range.")
    st.stop()

# ---------- Aggregations ----------
# Resample frequency map
freq_map = {"D": "D", "W": "W", "M": "M"}
agg_freq = freq_map[resample_freq]

agg = (dd := dff.set_index("date")).groupby([pd.Grouper(freq=agg_freq), "platform"]).agg(
    impressions=("impressions", "sum"),
    likes=("likes", "sum"),
    comments=("comments", "sum"),
    shares=("shares", "sum"),
    clicks=("clicks", "sum"),
    engagements=("engagements", "sum"),
    posts=("post_id", "nunique"),
    followers=("followers", "last")  # approximate follower at last entry in period
).reset_index().rename(columns={"date": "period"})

# Platform-level totals over selected range
platform_totals = dff.groupby("platform").agg(
    impressions=("impressions", "sum"),
    likes=("likes", "sum"),
    comments=("comments", "sum"),
    shares=("shares", "sum"),
    clicks=("clicks", "sum"),
    engagements=("engagements", "sum"),
    posts=("post_id", "nunique")
).reset_index()
platform_totals["engagement_rate"] = np.where(platform_totals["impressions"] > 0,
                                            platform_totals["engagements"] / platform_totals["impressions"], 0)
platform_totals["ctr"] = np.where(platform_totals["impressions"] > 0,
                                  platform_totals["clicks"] / platform_totals["impressions"], np.nan)

# Follower growth: for each platform compute (max followers - min followers)
if "followers" in dff.columns and dff["followers"].sum() > 0:
    follower_growth = dff.groupby("platform").agg(first_followers=("followers", "first"),
                                                  last_followers=("followers", "last")).reset_index()
    follower_growth["follower_growth"] = follower_growth["last_followers"] - follower_growth["first_followers"]
    platform_totals = platform_totals.merge(follower_growth[["platform", "first_followers", "last_followers", "follower_growth"]], on="platform", how="left")
else:
    platform_totals["first_followers"] = np.nan
    platform_totals["last_followers"] = np.nan
    platform_totals["follower_growth"] = np.nan

# ---------- KPI Tiles ----------
st.subheader("Platform Overview")
cols = st.columns(len(selected_platforms))
for i, plat in enumerate(selected_platforms):
    try:
        row = platform_totals[platform_totals["platform"] == plat].iloc[0]
    except IndexError:
        # if platform not present in totals (maybe filtered out), skip
        continue
    with cols[i % len(cols)]:
        st.metric(label=f"{plat} Impressions", value=int(row["impressions"]))
        st.metric(label=f"{plat} Engagement Rate", value=f"{row['engagement_rate']:.2%}")
        # show follower growth if available
        if not np.isnan(row.get("follower_growth", np.nan)):
            st.metric(label=f"{plat} Followers Δ", value=int(row["follower_growth"]))
        else:
            st.write("")  # spacer

st.markdown("---")

# ---------- Time-series charts ----------
st.subheader("Trends Over Time")
fig = px.line(agg, x="period", y="impressions", color="platform", markers=True,
              title=f"Impressions over time ({resample_freq})")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(agg, x="period", y="engagements", color="platform", markers=True,
               title=f"Engagements over time ({resample_freq})")
st.plotly_chart(fig2, use_container_width=True)

# ---------- Post type performance ----------
st.subheader("Post Type Performance")
ptype = dff.groupby("post_type").agg(
    impressions=("impressions", "sum"),
    engagements=("engagements", "sum"),
    posts=("post_id", "nunique")
).reset_index()
ptype["engagement_rate"] = np.where(ptype["impressions"] > 0, ptype["engagements"] / ptype["impressions"], 0)
fig3 = px.bar(ptype, x="post_type", y="engagement_rate", text=ptype["engagement_rate"].apply(lambda x: f"{x:.2%}"),
              title="Engagement rate by post type")
st.plotly_chart(fig3, use_container_width=True)

# ---------- Top posts ----------
st.subheader(f"Top {top_n} Posts by Engagement Rate")
top_posts = dff.assign(engagement_rate=lambda x: np.where(x["impressions"] > 0, x["engagements"] / x["impressions"], 0))
top_posts = top_posts.sort_values("engagement_rate", ascending=False).head(top_n)[
    ["platform", "date", "post_id", "post_type", "impressions", "likes", "comments", "shares", "engagement_rate", "caption"]
]
# Format engagement rate
top_posts["engagement_rate"] = top_posts["engagement_rate"].apply(lambda x: f"{x:.2%}")
st.dataframe(top_posts.reset_index(drop=True), use_container_width=True)

# ---------- Best posting days ----------
st.subheader("Best Posting Days (by engagement rate)")
dff["dayofweek"] = dff["date"].dt.day_name()
dow = dff.groupby("dayofweek").agg(engagements=("engagements", "sum"), impressions=("impressions", "sum")).reset_index()
dow["engagement_rate"] = np.where(dow["impressions"] > 0, dow["engagements"] / dow["impressions"], 0)
# order days
ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = dow.set_index("dayofweek").reindex(ordered_days).reset_index().fillna(0)
fig4 = px.bar(dow, x="dayofweek", y="engagement_rate", title="Engagement Rate by Day of Week",
              text=dow["engagement_rate"].apply(lambda x: f"{x:.2%}"))
st.plotly_chart(fig4, use_container_width=True)

# ---------- Auto insights (simple rules) ----------
st.markdown("---")
st.subheader("Automated Insights")
insights = []
# 1. Platform with highest engagement rate
best_platform = platform_totals.loc[platform_totals["engagement_rate"].idxmax()]
insights.append(f"Top performing platform (engagement rate): **{best_platform['platform']}** with {best_platform['engagement_rate']:.2%} engagement.")
# 2. Platform with most impressions
top_impr = platform_totals.loc[platform_totals["impressions"].idxmax()]
insights.append(f"Most impressions: **{top_impr['platform']}** with {int(top_impr['impressions'])} impressions.")
# 3. Best post type
best_pt = ptype.loc[ptype["engagement_rate"].idxmax()]
insights.append(f"Best post type by engagement rate: **{best_pt['post_type']}** ({best_pt['engagement_rate']:.2%}).")
# 4. Best day to post
best_day = dow.loc[dow["engagement_rate"].idxmax()]['dayofweek']
insights.append(f"Best day to post: **{best_day}** (highest average engagement rate).")
# 5. Follower growth observations
fg_available = platform_totals["follower_growth"].notna().any()
if fg_available:
    fg = platform_totals.loc[platform_totals["follower_growth"].idxmax()]
    if fg["follower_growth"] > 0:
        insights.append(f"Fastest follower growth: **{fg['platform']}** (+{int(fg['follower_growth'])} followers over the selected period).")
else:
    insights.append("Follower growth not available (no followers data).")

for i, insight in enumerate(insights, 1):
    st.markdown(f"{i}. {insight}")

# ---------- Exports ----------
st.markdown("---")
st.subheader("Export Data & Insights")

if export_agg:
    st.download_button("Download Aggregated Metrics (CSV)", data=agg.to_csv(index=False).encode("utf-8"),
                       file_name="social_aggregated_metrics.csv", mime="text/csv")

# Export platform totals
st.download_button("Download Platform Summary (CSV)", data=platform_totals.to_csv(index=False).encode("utf-8"),
                   file_name="social_platform_summary.csv", mime="text/csv")

# Export top posts
st.download_button("Download Top Posts (CSV)", data=top_posts.to_csv(index=False).encode("utf-8"),
                   file_name="social_top_posts.csv", mime="text/csv")

# Export simple text insights
insights_text = "\n".join([f"{i}. {ins}" for i, ins in enumerate(insights, 1)])
st.download_button("Download Insights (TXT)", data=insights_text.encode("utf-8"),
                   file_name="social_insights.txt", mime="text/plain")

st.markdown("---")
st.caption("Built for learning: extend by connecting to APIs (Facebook, Instagram, TikTok) and adding scheduling/alerts.")
