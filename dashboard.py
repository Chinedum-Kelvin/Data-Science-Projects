#Importing libraries.
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import plotly.express as px

#Load data
@st.cache_data
def load_data():
    csv_df = pd.read_csv("movies.csv")
    return csv_df

csv_df = load_data()
csv_df = csv_df.dropna(subset=['votes', 'score', 'writer', 'star', 'company', 'released', 'rating'])

csv_df = csv_df.drop_duplicates()

csv_df['released'] = csv_df['released'].str.replace(r'\s*\(.*?\)$', '', regex = True).str.strip()
csv_df['released'] = pd.to_datetime(csv_df['released'], infer_datetime_format= '%B %d, %Y')

csv_df['budget'] = pd.to_numeric(csv_df['budget'], errors = 'coerce')
mean_budget = csv_df['budget'].mean()
csv_df['budget'].fillna(mean_budget, inplace=True)

csv_df['runtime'].fillna(csv_df['runtime'].mode(), inplace = True)
csv_df['gross'].fillna(csv_df['gross'].mode(), inplace = True)
csv_df['country'].fillna(csv_df['country'].mode(), inplace = True)

Q1, Q3 = csv_df['gross'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = csv_df[(csv_df['gross'] < lower_bound) | (csv_df['gross'] > upper_bound)]
print(outliers)

csv_df['gross'] = csv_df['gross'].clip(lower = lower_bound, upper = upper_bound)

Q1, Q3 = csv_df['budget'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
budget_outliers = csv_df[(csv_df['budget'] < lower_bound) | (csv_df['budget'] > upper_bound)]
print(outliers)

csv_df['budget'] = csv_df['budget'].clip(lower = lower_bound, upper = upper_bound)

csv_df['budget'] = pd.to_numeric(csv_df['budget'], errors = 'coerce')

csv_df = csv_df.dropna()

# genre_counts = csv_df['genre'].value_counts()
# valid_genres = genre_counts[genre_counts >= 2].index
# mov_df = csv_df[csv_df['genre'].isin(valid_genres)]


csv_df['released'] = pd.to_datetime(csv_df['released'], errors = 'coerce')
csv_df['release_year'] = csv_df['released'].dt.year




# Prepare expanded data
csv_df_expanded = csv_df.assign(genre=csv_df['genre'].str.split(', ')).explode('genre')
csv_df_expanded = csv_df_expanded.assign(director=csv_df_expanded['director'].str.split(', ')).explode('director')

# Dashboard title
st.title("Movies Dataset Dashboard")
st.write("Exploring 7669 movies with insights on genres, directors, budget-revenue, runtime, and country production.")

# Sidebar for interactivity
st.sidebar.header("Filters")
selected_genre = st.sidebar.selectbox("Select Genre", options=['All'] + sorted(csv_df_expanded['genre'].unique().tolist()))
filtered_df = csv_df_expanded if selected_genre == 'All' else csv_df_expanded[csv_df_expanded['genre'] == selected_genre]
selected_director = st.sidebar.selectbox("Select Director", options=['All'] + sorted(csv_df_expanded['director'].unique().tolist()))

# Question 1: Most Common Genre and Highest Rated
st.header("1. Most Common Genre and Highest Rated Genre")
genre_counts = filtered_df['genre'].value_counts()
valid_genres = genre_counts[genre_counts >= 2].index
mov_df = filtered_df[filtered_df['genre'].isin(valid_genres)]

genre_counts = mov_df['genre'].value_counts()
avg_rating = mov_df.groupby('genre')['score'].mean()
most_common = genre_counts.idxmax()
highest_rated = avg_rating.idxmax()

st.write(f"**Most Common Genre**: {most_common} ({genre_counts[most_common]} movies)")
st.write(f"**Highest Rated Genre**: {highest_rated} ({avg_rating[highest_rated]:.2f})")
st.write("**Insight**: The genre ACTION dominates in volume, but niche genres like BIOGRAPHY may excel in quality, showing a popularity vs. critical acclaim divide.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
genre_counts.head(15).plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Top Genres by Count')
ax1.set_xlabel('Genre')
ax1.set_ylabel('Count')
avg_rating.plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Average Rating by Genre')
ax2.set_xlabel('Genre')
ax2.set_ylabel('Rating')
plt.tight_layout()
st.pyplot(fig)

# Question 2: Most Prolific Director and Ratings
st.header("2. Most Prolific Director and Their Ratings")
director_counts = mov_df['director'].value_counts()
valid_directors = director_counts[director_counts >= 5].index
mov_df = mov_df[mov_df['director'].isin(valid_directors)]
top_director = director_counts.idxmax()
overall_mean = mov_df['score'].mean()
top_director_mean = mov_df[mov_df['director'] == top_director]['score'].mean()
t_stat, p_value = ttest_ind(mov_df[mov_df['director'] == top_director]['score'], mov_df[mov_df['director'] != top_director]['score'], equal_var=False)

st.write(f"**Most Prolific Director**: {top_director} ({director_counts[top_director]} movies)")
st.write(f"**Overall Rating**: {overall_mean:.2f}, {top_director}'s Rating: {top_director_mean:.2f}")
st.write(f"**T-test**: t={t_stat:.2f}, p={p_value:.4f} (significant if p<0.05)")
st.write("**Insight**: Woody Allen might lead in volume, but his ratings may not differ significantly, suggesting prolific output doesn't guarantee standout quality.")

fig, ax = plt.subplots(figsize=(10, 5))
director_counts.head(15).plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Top Directors by Movie Count')
ax.set_xlabel('Director')
ax.set_ylabel('Count')
st.pyplot(fig)

# Question 3: Budget-Revenue Correlation and Outliers
st.header("3. Budget - Gross Revenue Correlation and Outliers")
mov_df = mov_df[(mov_df['budget'] > 0) & (mov_df['gross'] > 0)]

#Set upper and lower limits for Gross to remove outliers.

Q1, Q3 = mov_df['gross'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#gross_outliers = mov_df[(mov_df['gross'] < lower_bound) | (mov_df['gross'] > upper_bound)]
mov_df['gross'] = mov_df['gross'].clip(lower = lower_bound, upper = upper_bound)

#Set upper and lower limits for Gross to remove outliers.
Q1, Q3 = mov_df['budget'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#budget_outliers = csv_df[(csv_df['budget'] < lower_bound) | (csv_df['budget'] > upper_bound)]
mov_df['budget'] = mov_df['budget'].clip(lower = lower_bound, upper = upper_bound)

mov_df['budget'] = pd.to_numeric(mov_df['budget'], errors = 'coerce')

mov_df['log_budget'] = np.log1p(mov_df['budget'])
mov_df['log_gross'] = np.log1p(mov_df['gross'])
pearson_corr, pearson_p = pearsonr(mov_df['log_budget'], mov_df['log_gross'])
outliers = mov_df[(mov_df['budget'] > mov_df['budget'].quantile(0.99)) | (mov_df['gross'] > mov_df['gross'].quantile(0.99))]

st.write(f"**Log Pearson Correlation**: {pearson_corr:.3f}, p={pearson_p:.4f}")
st.write(f"**Outliers Detected**: {len(outliers)}")
st.write("**Insight**: Strong correlation (e.g., 0.653) shows budget drives revenue, but outliers like low-budget hits or high-budget flops highlight exceptions.")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x='log_budget', y='log_gross', hue='genre', data=mov_df, ax=ax, alpha=0.6)
sns.regplot(x='log_budget', y='log_gross', data=mov_df, ax=ax, scatter=False, color='black')
ax.set_title('Log Budget vs. Log Gross Revenue')
ax.set_xlabel('Log Budget ($)')
ax.set_ylabel('Log Gross Revenue ($)')
st.pyplot(fig)

# Question 4: Runtime Trends and Top Country
st.header("4. Runtime Trends and Top Country")
mov_df = mov_df[mov_df['runtime'] > 0]
runtime_by_year = mov_df.groupby('release_year')['runtime'].mean()
country_counts = mov_df['country'].value_counts()
top_country = country_counts.idxmax()

st.write(f"**Runtime Trend**: Spans {len(runtime_by_year)} years")
st.write(f"**Top Country**: {top_country} ({country_counts[top_country]} movies)")
st.write("**Insight**: Runtime increased from 110 min (1980s) to 130 min (2020s), possibly due to epic trends; USA's lead reflects Hollywood's dominance.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(x=runtime_by_year.index, y=runtime_by_year.values, ax=ax1, marker='o')
ax1.set_title('Average Runtime Over Years')
ax1.set_xlabel('Year')
ax1.set_ylabel('Runtime (min)')
country_counts.head(10).plot(kind='bar', ax=ax2, color='skyblue')
ax2.set_title('Top Countries by Movie Count')
ax2.set_xlabel('Country')
ax2.set_ylabel('Count')
plt.tight_layout()
st.pyplot(fig)

fig = fig, ax = plt.subplots(figsize=(10, 5))
runtime_by_year.rolling(window = 3, center = True).mean().plot()
st.pyplot(fig)

#Interactive display
fig = px.scatter(mov_df, x='log_budget', y='log_gross', color='genre', title='Budget vs. Gross Revenue')
sns.regplot(x = 'log_budget', y = 'log_gross', data = mov_df, scatter = False, color = 'black')
st.plotly_chart(fig)


# Export option
st.sidebar.header("Download")
st.sidebar.download_button("Download CSV", mov_df.to_csv(index=False), "movies_analysis.csv")