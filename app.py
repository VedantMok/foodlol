from pathlib import Path
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(
    page_title="Restaurant Ratings & Satisfaction Dashboard",
    page_icon="🍽️",
    layout="wide",
)

DATA_FILE = "restaurant_kpi_dashboard_dataset.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["snapshot_month"] = pd.to_datetime(df["snapshot_month"])
    numeric_cols = [
        "price_range", "avg_cost_for_two_usd", "seating_capacity", "total_orders",
        "dine_in_orders", "delivery_orders", "reservations_count", "unique_customers",
        "new_customers", "repeat_customers", "repeat_customer_rate_pct", "total_reviews",
        "avg_review_length_words", "negative_review_share_pct", "avg_food_rating",
        "avg_service_rating", "avg_ambience_rating", "avg_value_rating",
        "avg_cleanliness_rating", "overall_rating", "customer_satisfaction_score", "nps",
        "avg_wait_time_min", "avg_delivery_time_min", "on_time_delivery_rate_pct",
        "complaint_rate_pct", "discount_share_pct", "ad_spend_usd", "revenue_usd",
        "revenue_per_customer_usd", "month_over_month_revenue_growth_pct"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month_label"] = df["snapshot_month"].dt.strftime("%Y-%m")
    return df


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if valid.sum() == 0 or weights[valid].sum() == 0:
        return np.nan
    return np.average(values[valid], weights=weights[valid])


def build_kpi_summary(df: pd.DataFrame):
    weighted_rating = weighted_average(df["overall_rating"], df["total_reviews"].clip(lower=1))
    avg_satisfaction = df["customer_satisfaction_score"].mean()
    avg_reviews = df["total_reviews"].mean()
    avg_repeat = df["repeat_customer_rate_pct"].mean()
    total_revenue = df["revenue_usd"].sum()
    return weighted_rating, avg_satisfaction, avg_reviews, avg_repeat, total_revenue


def driver_importance(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["avg_food_rating", "avg_service_rating", "avg_ambience_rating", "avg_value_rating"]
    base = df[cols + ["overall_rating"]].dropna().copy()
    X = base[cols]
    y = base["overall_rating"]
    Xs = (X - X.mean()) / X.std(ddof=0)
    ys = (y - y.mean()) / y.std(ddof=0)
    Xmat = np.column_stack([np.ones(len(Xs)), Xs.values])
    beta = np.linalg.lstsq(Xmat, ys.values, rcond=None)[0][1:]
    imp = pd.DataFrame({
        "driver": ["Food", "Service", "Ambience", "Value"],
        "importance": np.abs(beta),
        "direction": np.where(beta >= 0, "Positive", "Negative")
    }).sort_values("importance", ascending=False)
    imp["importance_label"] = imp["importance"].round(3)
    return imp


def city_cuisine_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["city", "cuisine_primary"], as_index=False)
          .apply(lambda x: pd.Series({
              "weighted_rating": weighted_average(x["overall_rating"], x["total_reviews"].clip(lower=1)),
              "reviews": x["total_reviews"].sum(),
              "satisfaction": x["customer_satisfaction_score"].mean()
          }))
          .reset_index(drop=True)
    )
    grouped["tooltip_reviews"] = grouped["reviews"].round(0).astype(int)
    return grouped


def cuisine_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("cuisine_primary", as_index=False)
          .apply(lambda x: pd.Series({
              "weighted_rating": weighted_average(x["overall_rating"], x["total_reviews"].clip(lower=1)),
              "avg_satisfaction": x["customer_satisfaction_score"].mean(),
              "total_reviews": x["total_reviews"].sum(),
              "restaurants": x["restaurant_id"].nunique()
          }))
          .reset_index(drop=True)
          .sort_values(["weighted_rating", "avg_satisfaction"], ascending=False)
    )
    return out


def city_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("city", as_index=False)
          .apply(lambda x: pd.Series({
              "weighted_rating": weighted_average(x["overall_rating"], x["total_reviews"].clip(lower=1)),
              "avg_satisfaction": x["customer_satisfaction_score"].mean(),
              "avg_reviews": x["total_reviews"].mean(),
              "complaint_rate": x["complaint_rate_pct"].mean(),
              "restaurants": x["restaurant_id"].nunique()
          }))
          .reset_index(drop=True)
          .sort_values("weighted_rating", ascending=False)
    )
    return out


def impact_gap(df: pd.DataFrame) -> pd.DataFrame:
    city_avg = df.groupby("city")["overall_rating"].mean().rename("city_avg_rating")
    x = df.merge(city_avg, on="city", how="left")
    x["rating_gap_vs_city"] = x["overall_rating"] - x["city_avg_rating"]
    x["service_minus_food"] = x["avg_service_rating"] - x["avg_food_rating"]
    x["issue_flag"] = np.where(
        (x["rating_gap_vs_city"] < -0.12) & (x["avg_service_rating"] < x["avg_food_rating"]),
        "Service drag",
        np.where(x["rating_gap_vs_city"] < -0.12, "Underperforming", "Normal/Strong")
    )
    return x


def make_download(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


if not Path(DATA_FILE).exists():
    st.error(f"Missing data file: {DATA_FILE}. Add it to the repository root before deploying.")
    st.stop()

source = load_data(DATA_FILE)

st.title("Restaurant Ratings & Satisfaction Dashboard")
st.caption("Focused on descriptive and diagnostic analytics to understand what influences restaurant ratings and customer satisfaction.")

with st.sidebar:
    st.header("Filters")
    city_options = sorted(source["city"].dropna().unique().tolist())
    cuisine_options = sorted(source["cuisine_primary"].dropna().unique().tolist())
    type_options = sorted(source["restaurant_type"].dropna().unique().tolist())
    month_min = source["snapshot_month"].min().date()
    month_max = source["snapshot_month"].max().date()

    selected_cities = st.multiselect("City", city_options, default=city_options)
    selected_cuisines = st.multiselect("Cuisine", cuisine_options, default=cuisine_options)
    selected_types = st.multiselect("Restaurant type", type_options, default=type_options)
    selected_prices = st.multiselect("Price range", sorted(source["price_range"].dropna().unique().tolist()), default=sorted(source["price_range"].dropna().unique().tolist()))
    selected_months = st.slider("Month range", min_value=month_min, max_value=month_max, value=(month_min, month_max))

filtered = source[
    source["city"].isin(selected_cities)
    & source["cuisine_primary"].isin(selected_cuisines)
    & source["restaurant_type"].isin(selected_types)
    & source["price_range"].isin(selected_prices)
    & source["snapshot_month"].between(pd.to_datetime(selected_months[0]), pd.to_datetime(selected_months[1]))
].copy()

if filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()

weighted_rating, avg_satisfaction, avg_reviews, avg_repeat, total_revenue = build_kpi_summary(filtered)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Weighted overall rating", f"{weighted_rating:.2f}")
m2.metric("Avg satisfaction", f"{avg_satisfaction:.1f}")
m3.metric("Avg reviews / record", f"{avg_reviews:.1f}")
m4.metric("Repeat customer rate", f"{avg_repeat:.1f}%")
m5.metric("Revenue", f"${total_revenue:,.0f}")

trend_df = (
    filtered.groupby("snapshot_month", as_index=False)
    .agg(
        overall_rating=("overall_rating", "mean"),
        satisfaction=("customer_satisfaction_score", "mean"),
        total_reviews=("total_reviews", "sum")
    )
)

trend_long = trend_df.melt("snapshot_month", var_name="metric", value_name="value")
trend_chart = alt.Chart(trend_long).mark_line(point=True, strokeWidth=3).encode(
    x=alt.X("snapshot_month:T", title="Month"),
    y=alt.Y("value:Q", title="Metric value"),
    color=alt.Color("metric:N", title="Metric"),
    tooltip=[
        alt.Tooltip("snapshot_month:T", title="Month"),
        alt.Tooltip("metric:N", title="Metric"),
        alt.Tooltip("value:Q", title="Value", format=",.2f")
    ]
).properties(height=320, title="Performance trend")

st.altair_chart(trend_chart, use_container_width=True)

cuisine_df = cuisine_leaderboard(filtered)
city_df = city_benchmark(filtered)
heat_df = city_cuisine_heatmap(filtered)
impact_df = impact_gap(filtered)
driver_df = driver_importance(filtered)

# Tabs
intro_tab, desc_tab, diag_tab, pred_tab, pres_tab = st.tabs([
    "Overview", "Descriptive", "Diagnostic", "Predictive Later", "Prescriptive Later"
])

with intro_tab:
    c1, c2 = st.columns([1.15, 1])

    cuisine_bar = alt.Chart(cuisine_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("weighted_rating:Q", title="Review-weighted rating", scale=alt.Scale(domain=[3, 5])),
        y=alt.Y("cuisine_primary:N", sort="-x", title="Cuisine"),
        color=alt.Color("avg_satisfaction:Q", title="Avg satisfaction", scale=alt.Scale(scheme="goldgreen")),
        tooltip=[
            alt.Tooltip("cuisine_primary:N", title="Cuisine"),
            alt.Tooltip("weighted_rating:Q", title="Weighted rating", format=",.2f"),
            alt.Tooltip("avg_satisfaction:Q", title="Avg satisfaction", format=",.1f"),
            alt.Tooltip("total_reviews:Q", title="Reviews", format=",.0f"),
            alt.Tooltip("restaurants:Q", title="Restaurants", format=",.0f")
        ]
    ).properties(height=420, title="Cuisine leaderboard")

    city_dot = alt.Chart(city_df).mark_circle(size=220, opacity=0.85).encode(
        x=alt.X("avg_reviews:Q", title="Average reviews per record"),
        y=alt.Y("weighted_rating:Q", title="Weighted rating", scale=alt.Scale(domain=[3, 5])),
        size=alt.Size("restaurants:Q", title="Restaurant count", scale=alt.Scale(range=[150, 1400])),
        color=alt.Color("complaint_rate:Q", title="Complaint rate %", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
        tooltip=[
            alt.Tooltip("city:N", title="City"),
            alt.Tooltip("weighted_rating:Q", title="Weighted rating", format=",.2f"),
            alt.Tooltip("avg_satisfaction:Q", title="Avg satisfaction", format=",.1f"),
            alt.Tooltip("avg_reviews:Q", title="Avg reviews", format=",.1f"),
            alt.Tooltip("complaint_rate:Q", title="Complaint rate", format=",.2f"),
            alt.Tooltip("restaurants:Q", title="Restaurants", format=",.0f")
        ]
    ).properties(height=420, title="City benchmark map")

    c1.altair_chart(cuisine_bar, use_container_width=True)
    c2.altair_chart(city_dot, use_container_width=True)

    st.download_button(
        "Download filtered data",
        data=make_download(filtered),
        file_name="filtered_restaurant_dashboard_data.csv",
        mime="text/csv"
    )

with desc_tab:
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    price_box = alt.Chart(filtered).mark_boxplot(extent="min-max", size=30).encode(
        x=alt.X("price_range:O", title="Price range"),
        y=alt.Y("overall_rating:Q", title="Overall rating", scale=alt.Scale(domain=[2.5, 5])),
        color=alt.Color("price_range:O", legend=None)
    ).properties(height=350, title="Price range vs rating distribution")

    review_points = alt.Chart(filtered).mark_circle(size=85, opacity=0.6).encode(
        x=alt.X("total_reviews:Q", title="Number of reviews"),
        y=alt.Y("overall_rating:Q", title="Overall rating", scale=alt.Scale(domain=[2.5, 5])),
        color=alt.Color("city:N", title="City"),
        size=alt.Size("revenue_usd:Q", title="Revenue", scale=alt.Scale(range=[60, 700])),
        tooltip=[
            "restaurant_name:N", "city:N", "cuisine_primary:N",
            alt.Tooltip("price_range:Q", title="Price range"),
            alt.Tooltip("total_reviews:Q", title="Reviews", format=",.0f"),
            alt.Tooltip("overall_rating:Q", title="Overall rating", format=",.2f"),
            alt.Tooltip("customer_satisfaction_score:Q", title="Satisfaction", format=",.1f")
        ]
    )
    review_reg = review_points.transform_regression("total_reviews", "overall_rating").mark_line(color="#111827", strokeDash=[6,4], size=3)
    review_chart = (review_points + review_reg).properties(height=350, title="Reviews vs overall rating")

    heatmap = alt.Chart(heat_df).mark_rect().encode(
        x=alt.X("city:N", title="City"),
        y=alt.Y("cuisine_primary:N", title="Cuisine", sort="-color"),
        color=alt.Color("weighted_rating:Q", title="Weighted rating", scale=alt.Scale(scheme="tealblues", domain=[3, 5])),
        tooltip=[
            "city:N", "cuisine_primary:N",
            alt.Tooltip("weighted_rating:Q", title="Weighted rating", format=",.2f"),
            alt.Tooltip("satisfaction:Q", title="Avg satisfaction", format=",.1f"),
            alt.Tooltip("tooltip_reviews:Q", title="Reviews", format=",.0f")
        ]
    ).properties(height=380, title="City-cuisine rating heatmap")

    city_line = alt.Chart(filtered).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("snapshot_month:T", title="Month"),
        y=alt.Y("mean(overall_rating):Q", title="Average rating", scale=alt.Scale(domain=[3, 5])),
        color=alt.Color("city:N", title="City"),
        tooltip=[alt.Tooltip("snapshot_month:T", title="Month"), alt.Tooltip("mean(overall_rating):Q", title="Avg rating", format=",.2f")]
    ).properties(height=380, title="City rating trend")

    top_left.altair_chart(price_box, use_container_width=True)
    top_right.altair_chart(review_chart, use_container_width=True)
    bottom_left.altair_chart(heatmap, use_container_width=True)
    bottom_right.altair_chart(city_line, use_container_width=True)

with diag_tab:
    left, right = st.columns(2)
    bottom_l, bottom_r = st.columns(2)

    driver_bar = alt.Chart(driver_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("importance:Q", title="Standardized impact on overall rating"),
        y=alt.Y("driver:N", sort="-x", title="Driver"),
        color=alt.Color("direction:N", title="Direction", scale=alt.Scale(domain=["Positive", "Negative"], range=["#059669", "#DC2626"])),
        tooltip=[
            alt.Tooltip("driver:N", title="Driver"),
            alt.Tooltip("importance:Q", title="Impact", format=",.3f"),
            alt.Tooltip("direction:N", title="Direction")
        ]
    ).properties(height=350, title="Aspect driver importance")

    diagnostic_scatter = alt.Chart(impact_df).mark_circle(size=95, opacity=0.68).encode(
        x=alt.X("avg_service_rating:Q", title="Service rating"),
        y=alt.Y("overall_rating:Q", title="Overall rating", scale=alt.Scale(domain=[2.5, 5])),
        color=alt.Color("issue_flag:N", title="Flag", scale=alt.Scale(domain=["Service drag", "Underperforming", "Normal/Strong"], range=["#DC2626", "#F59E0B", "#10B981"])),
        size=alt.Size("complaint_rate_pct:Q", title="Complaint rate %", scale=alt.Scale(range=[80, 850])),
        tooltip=[
            "restaurant_name:N", "city:N", "cuisine_primary:N",
            alt.Tooltip("avg_food_rating:Q", title="Food rating", format=",.2f"),
            alt.Tooltip("avg_service_rating:Q", title="Service rating", format=",.2f"),
            alt.Tooltip("avg_ambience_rating:Q", title="Ambience rating", format=",.2f"),
            alt.Tooltip("overall_rating:Q", title="Overall rating", format=",.2f"),
            alt.Tooltip("complaint_rate_pct:Q", title="Complaint rate %", format=",.2f"),
            "issue_flag:N"
        ]
    ).properties(height=350, title="Service drag detector")

    melt_cols = ["avg_food_rating", "avg_service_rating", "avg_ambience_rating", "avg_value_rating", "overall_rating"]
    corr = filtered[melt_cols].corr().stack().reset_index()
    corr.columns = ["metric_1", "metric_2", "correlation"]
    corr_heat = alt.Chart(corr).mark_rect().encode(
        x=alt.X("metric_1:N", title=None),
        y=alt.Y("metric_2:N", title=None),
        color=alt.Color("correlation:Q", title="Correlation", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
        tooltip=["metric_1:N", "metric_2:N", alt.Tooltip("correlation:Q", format=",.2f")]
    ).properties(height=360, title="Rating driver correlation matrix")

    issue_counts = (
        impact_df.groupby("recommended_action", as_index=False)
        .agg(restaurants=("restaurant_id", "nunique"), avg_rating=("overall_rating", "mean"), avg_satisfaction=("customer_satisfaction_score", "mean"))
        .sort_values("restaurants", ascending=False)
    )
    action_bar = alt.Chart(issue_counts).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("restaurants:Q", title="Restaurant count"),
        y=alt.Y("recommended_action:N", sort="-x", title="Recommended action"),
        color=alt.Color("avg_rating:Q", title="Avg rating", scale=alt.Scale(scheme="orangered")),
        tooltip=[
            "recommended_action:N",
            alt.Tooltip("restaurants:Q", title="Restaurants", format=",.0f"),
            alt.Tooltip("avg_rating:Q", title="Avg rating", format=",.2f"),
            alt.Tooltip("avg_satisfaction:Q", title="Avg satisfaction", format=",.1f")
        ]
    ).properties(height=360, title="Operational action mix")

    left.altair_chart(driver_bar, use_container_width=True)
    right.altair_chart(diagnostic_scatter, use_container_width=True)
    bottom_l.altair_chart(corr_heat, use_container_width=True)
    bottom_r.altair_chart(action_bar, use_container_width=True)

    restaurant_detail = impact_df[[
        "month_label", "restaurant_name", "city", "cuisine_primary", "price_range", "total_reviews",
        "avg_food_rating", "avg_service_rating", "avg_ambience_rating", "overall_rating",
        "customer_satisfaction_score", "complaint_rate_pct", "recommended_action", "issue_flag"
    ]].sort_values(["overall_rating", "customer_satisfaction_score"], ascending=[True, True])
    st.dataframe(restaurant_detail, use_container_width=True, hide_index=True)

with pred_tab:
    st.info("Placeholder: later you can add rating prediction, satisfaction risk scoring, CLV-style customer value, and next-period performance forecasting.")
    st.markdown("Suggested next models: overall rating regression, satisfaction classification, review growth forecast, and city-cuisine demand prediction.")

with pres_tab:
    st.info("Placeholder: later you can add what-if simulations, intervention prioritization, and experiment tracking.")
    st.markdown("Suggested actions: pricing tests, staffing improvement experiments, targeted promotions, and A/B testing by city or cuisine.")
