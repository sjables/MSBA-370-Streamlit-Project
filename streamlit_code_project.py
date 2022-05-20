# --------------- Import packages
import streamlit as st
import pandas as pd
import numpy as np
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2, country_name_to_country_alpha3, convert_continent_code_to_continent_name
# scaling 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# PCA
from sklearn.decomposition import PCA
# kmeans clustering 
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
# Visualization tools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "notebook_connected"

# ----------------- Start
@st.cache
# Set title
st.title('Clustering Humanitarian Aid need levels based on UNDP MPI factors')

##### Sidebar
with st.sidebar:
    st.header("""Navigate this project: """)

page = st.sidebar.selectbox("", ["🏠 Home", "📊 Explore the dataset", "💡 Cluster", "🤖 ML code and scores"])
st.sidebar.markdown("""---""")

# Upload file 
uploaded_file = st.sidebar.file_uploader("Choose and upload an MPI Excel file", type="xlsx") #data uploader

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

else:
    st.markdown("""
    <br>
    <h3 style="color:#800000;"> 🚨 Upload your Excel file to begin clustering </h3>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>

    """, unsafe_allow_html=True)

# ------------------------------------------------- Backend (for your eyes only)

# ---------------- 1. Data cleaning and basic visualization

## Hidden: Create a country ISO-3 column 
def country_to_ISO3(Country):
    try:
        country_alpha3 = country_name_to_country_alpha3(Country)
    except:
        country_alpha3 = 'Unknown'
    return country_alpha3

df['ISO3'] = df.apply(lambda row: country_to_ISO3(row.Country), axis = 1)

# Create a continent column 
def country_to_continent(Country):
    try:
        country_alpha2 = country_name_to_country_alpha2(Country)
    except:
        country_alpha2 = 'Unknown'
    try:
        country_continent_code = country_alpha2_to_continent_code(country_alpha2)
    except:
        country_continent_code = 'Unknown' 
    try:
        country_continent_name = convert_continent_code_to_continent_name(country_continent_code)
    except:
        country_continent_name = 'Unknown'
    return country_continent_name
# Append continent column on original dataframe
df['Continent'] = df.apply(lambda row: country_to_continent(row.Country), axis = 1)

# Manually change continents that function failed to identify
df.iloc[[15], 15] = 'Africa'
df.iloc[[16], 15] = 'Africa'
df.iloc[[75], 15] = 'Asia'
df.iloc[[86], 15] = 'South America'
# Manually change ISO3 codes that function failed to identify
df.iloc[[15], 14] = 'COG'
df.iloc[[16], 14] = 'COD'
df.iloc[[86], 14] = 'BOL'

# Show distributions of all features
df.hist(bins=50, figsize=(20,15))
plt.show()

# Correlation matrix of all features
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix,dtype = bool))
plt.figure(dpi=100)
ax = plt.axes()
sns.heatmap(corr_matrix,annot=False, mask=mask,lw=0,linecolor='white',fmt = "0.2f") # Plot correlation matrix
plt.title('Correlation Analysis')
plt.show()

# Drop null values and categorical columns
df.dropna(inplace=True)
df2 = df.drop(['Country', 'Continent', 'ISO3'], axis =1)
df2.head()

# Visualize pairwise relationships
sns.pairplot(df2 ,corner=True, diag_kind="kde")
plt.show()

# ---------------- 2. Preprocessing

# Normalization using MinMax 
columns = df2.columns # Will be used to add back to original dataframe (df)
scaler = MinMaxScaler() # Used to convert values to a range between 0 and 1
scaled_df2_minmax = scaler.fit_transform(df2) # Apply Normalizer to dataframe

# Standardization using StandardScaler
columns = df2.columns # Will be used to add back to original dataframe (df)
scaler = StandardScaler() # Used to convert mean to 0 and standard deviation to 1
scaled_df2_ss = scaler.fit_transform(df2) # Apply Standardizer to dataframe

# Convert both to dataframes
df_minmax = pd.DataFrame(data= scaled_df2_minmax , columns = columns)
df_ss = pd.DataFrame(data= scaled_df2_ss , columns = columns)

# fit and transform PCA to choose optimal PCAs (unsupervised ML doesn't require splitting)
# 1. Normalized dataset
pca = PCA()
pca.fit(df_minmax)
pca_data_minmax = pca.transform(df_minmax)

# percentage variation 
per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
labels = ['PC' + str(x) for x in range (1, len(per_var)+1)]

# plot the percentage of explained variance by principal component
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# plot pca
pca_df_minmax = pd.DataFrame(pca_data_minmax, columns = labels)
plt.scatter(pca_df_minmax.PC1, pca_df_minmax.PC2)
plt.title('PCA')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

# fit and transform PCA to choose optimal PCAs (unsupervised ML doesn't require splitting)
# 1. Standardized dataset
pca = PCA()
pca.fit(df_ss)
pca_data_standard = pca.transform(df_ss)

# percentage variation 
per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
labels = ['PC' + str(x) for x in range (1, len(per_var)+1)]

# plot the percentage of explained variance by principal component
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# plot pca
pca_df_standard = pd.DataFrame(pca_data_standard, columns = labels)
plt.scatter(pca_df_standard.PC1, pca_df_standard.PC2)
plt.title('PCA')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

# Based on visualization output, drop PCAs that don't contribute as much
df_ss_pca = pca_df_standard.drop(['PC6','PC7','PC8','PC9','PC10', 'PC11', 'PC12', 'PC13'], axis = 1)

# Establish model parameters
km = KMeans (
    n_clusters = 3,
    init = 'random',
    n_init = 10,
    max_iter = 300,
    tol = 1e-4,
    random_state = 0
)

# Fit K Means model on all three datasets (normalized, standardized, and dataset with selected PCAs)
y_predicted_minmax = km.fit_predict(df_minmax)
y_predicted_ss = km.fit_predict(df_ss)
y_predicted_pca_ss = km.fit_predict(df_ss_pca)

# Add cluster column to normalized dataset
df_minmax['cluster'] = y_predicted_minmax
df_minmax.head()

# Add cluster column to standardized dataset
df_ss['cluster'] = y_predicted_ss
df_ss.head()

# Add cluster column to numerical variables dataframe (we will merge this with categorical columns eventually)
df2['cluster'] = y_predicted_pca_ss
df2.head()

# Optimal number of clusters (Elbow method)
# For normalized dataframe
sse = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(df_minmax)
    sse.append(km.inertia_)

# plot
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# For standardized dataframe
sse = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(df_ss)
    sse.append(km.inertia_)

# plot
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# For selected PCAs dataframe
sse = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(df2)
    sse.append(km.inertia_)

# plot
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Optimal number of clusters (silhouette method) -- Evaluate! 
# Print silhouette scores

# Normalized dataframe
score_mm = silhouette_score(df_minmax, km.labels_, metric='manhattan')
print('Silhouette Score: %.3f' % score_mm)
# Standardized dataframe
score_ss = silhouette_score(df_ss, km.labels_, metric='manhattan')
print('Silhouette Score: %.3f' % score_ss)
# Selected PCAs dataframe
score_df2 = silhouette_score(df2, km.labels_, metric='manhattan')
print('Silhouette Score: %.3f' % score_df2)

# Visualize Silhouettes

# Normalized dataframe
fig,ax = plt.subplots(2,2, figsize = (15,8))
for i in [2,3,4,5]:

    # create kmeans instance for different numbers of clusters
    km = KMeans(n_clusters=i, init= 'random', n_init =10, max_iter = 300, random_state = 0)
    q, mod = divmod(i,2)
    
    #create visualiser
    visualizer1 = SilhouetteVisualizer(km, colors = 'yellowbrick', ax=ax[q-1][mod])
    visualizer1.fit(df_minmax)

# Standardized dataframe
fig,ax = plt.subplots(2,2, figsize = (15,8))
for i in [2,3,4,5]:

    # create kmeans instance for different numbers of clusters
    km = KMeans(n_clusters=i, init= 'random', n_init =10, max_iter = 300, random_state = 0)
    q, mod = divmod(i,2)
    
    #create visualiser
    visualizer2 = SilhouetteVisualizer(km, colors = 'yellowbrick', ax=ax[q-1][mod])
    visualizer2.fit(df_ss)

# Selected PCAs dataframe
fig_ss,ax = plt.subplots(2,2, figsize = (15,8))
for i in [2,3,4,5]:

    # create kmeans instance for different numbers of clusters
    km = KMeans(n_clusters=i, init= 'random', n_init =10, max_iter = 300, random_state = 0)
    q, mod = divmod(i,2)
    
    #create visualiser
    visualizer3 = SilhouetteVisualizer(km, colors = 'yellowbrick', ax=ax[q-1][mod])
    visualizer3.fit(df2)

## Optimal dataframe is actually the one with selected PCAs, as shown in the silhouette visualizer and based on Silhouette Scores

# Merge selected dataframe (with selected PCAs) with categorical features
df_merged = pd.merge(df2, df[['Country','Continent', 'ISO3', 'MPI Index']], on='MPI Index', how='inner')

# ------------------------------------------------- Streamlit! 
if page == '🏠 Home':
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<span style="word-wrap:break-word;">The purpose of this project is to categorize countries listed in the 🇺🇳 [UNDP MPI dataset](https://hdr.undp.org/en/2021-MPI) based on their 'poverty intensity', measured by the three poverty dimensions- Health, Education, and Standard of Living. This can make it eaiser for International NGOs to allocate funding based on urgency and intensity of each of these countries. The machine learning component allows the user to upload an updated version of the UNDP MPI dataset (updated every few years) to track any changes in the MPI intensity of each country.</span>""", unsafe_allow_html=True)
    with col2:
        st.image('https://images.unsplash.com/photo-1561976167-fa9e36b104ef?ixlib=rb-1.2.1&raw_url=true&q=80&fm=jpg&crop=entropy&cs=tinysrgb&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870', caption='Photo by Jon Tyson on Unsplash')

    st.markdown("""
    <h3 style="color:#006400;"> ✅ Start exploring through the navigation panel on the left </h3>
    """, unsafe_allow_html=True)

elif page == '📊 Explore the dataset':
    container1 = st.container()
    container1.header('Explore the raw UNDP dataset')
    # Show raw data
    container1.write(df)
    container1.caption("""Above is a preview of our raw dataset, which will be cleaned and processed, visualized (below), and inputted into our model.""")

    # Data exploration 
    # 1. Distribution of MPI score
    container2 = st.container()
    mpi_dist = px.histogram(df, x="MPI Index", color_discrete_sequence=['indianred'])
    mpi_dist.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    mpi_dist.update_layout(
        autosize=False,
        margin = dict(
                l=10,
                r=10,
                b=10,
                t=10,
                pad=4,
                autoexpand=True),
                width=700)
    container2.write("""___________________________________________""")
    container2.write("""Multi-Poverty Index distribution""")
    container2.plotly_chart(mpi_dist)
    container2.caption("""The above chart shows a distribution of each country's MPI index.""")

    # 2. # Map showing all countries' MPI index intensity
    container3 = st.container()
    mpi_map = go.Figure(px.choropleth(df,
    locations = 'ISO3',
    color=df["MPI Index"].astype(float),
    color_continuous_scale="Brwnyl", 
    hover_name='Country',
    locationmode='ISO-3',
    height=600
    ))
    mpi_map.update_layout(mapbox_style="carto-positron")
    mpi_map.update_layout(
        autosize=False,
        margin = dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=4,
                autoexpand=True
            ),
            width=800
    )
    container3.write("""___________________________________________""")
    container3.write("""MPI levels acorss the world""")
    container3.plotly_chart(mpi_map)
    container3.caption("""The above map displays each country's MPI based on its intensity (darker = higher MPI)""")

    # Proportion of MPI dimensions in each continent
    container4 = st.container()
    continent = df['Continent']
    dim_prop_cont = go.Figure(data=[
        go.Bar(name='Health', x=continent, y=df['Contribution - Health (percentage)']),
        go.Bar(name='Education', x=continent, y=df['Contribution - Education (percentage)']),
        go.Bar(name='Standard of Living', x=continent, y=df['Contribution - Standard of living (percentage)'])
    ])
    # Change the bar mode
    dim_prop_cont.update_layout(barmode='stack')
    dim_prop_cont.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    dim_prop_cont.update_layout(
    autosize=False,
    margin = dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=4,
            autoexpand=True),
        width=800)
    container4.write("""___________________________________________""")
    container4.write("""MPI Contribution proportions per continent""")
    container4.plotly_chart(dim_prop_cont)
    container4.caption("""The above stacked barchart displays the three MPI proportions across each continent""")

elif page == '💡 Cluster':
    container5 = st.container()
    container5.header('K-Means Model Output')
    container5.write("""The below output visualizes each country and its respective cluster, based on an unsupervised machine learning algorithm that finds the optimal number of clusters and assigns each country to a cluster accordingly.""")

    # Map showing all countries and their clusters
    mpi_map2 = go.Figure(px.choropleth(df_merged,
    locations = 'ISO3',
    color=df_merged["cluster"],
    hover_name='Country',
    locationmode='ISO-3',
    title='Clusters around the world',
    height=600
    ))
    mpi_map2.update_layout(mapbox_style="carto-positron")
    mpi_map2.update_layout(
        autosize=False,
        margin = dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=4,
            autoexpand=True),
            width=800)
    container5.plotly_chart(mpi_map2)

    container5.write("""Takeaways*: """)
    container5.markdown("""**Cluster 0** countries, while still characterized by the MPI, are in less need of assistance as clusters 1 and 2.""")
    container5.markdown("""**Cluster 1**, while still better off than cluster 2, requires international aid on some level, as characterized by the poverty dimensions based on the database. While better off than cluster 2, they are in need of aid with one or two of the three dimensions, depending on the country and its characterization.""")
    container5.markdown("""**Cluster 2** is in dire need of international aid, as it is ranked as the most severly affected by poverty; classified by health, education, and living standards.""")
    container5.write(""" *_subject to change upon change in dataset_""")

else: 
    option = st.selectbox('Do you want to look at the ML code or the model performance?', ('Nitty gritty code', 'Model Performance'))
    if option == 'Nitty gritty code': 
        code = '''
        # Establish model parameters
        km = KMeans(
            n_clusters = 3,
            init = 'random',
            n_init = 10,
            max_iter = 300,
            tol = 1e-4,
            random_state = 0)
        
        # Predict
        y_predicted_pca_ss = km.fit_predict(df_ss_pca)
        df2['cluster'] = y_predicted_pca_ss
        df2.head()

        # Elbow method
        sse = []
        for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0)
        km.fit(df2)
        sse.append(km.inertia_)

        # Evaluate! Plot Elbow Method
        plt.plot(range(1, 11), sse, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.show()

        # Evaluate! Silhouette Score 
        score_df2 = silhouette_score(df2, km.labels_, metric='manhattan')
        print('Silhouette Score: %.3f' % score_df2)

        # Visualize Silhouette
        fig,ax = plt.subplots(2,2, figsize = (15,8))
        for i in [2,3,4,5]:
        # create kmeans instance for different numbers of clusters
        km = KMeans(n_clusters=i, init= 'random', n_init =10, max_iter = 300, random_state = 0)
        q, mod = divmod(i,2)
        #create visualiser
        visualizer = SilhouetteVisualizer(km, colors = 'yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(df2)
        '''
        st.code(code, language='python')
    if option == 'Model Performance':
        st.write("""The below dataframe is the finalized dataset with the corresponding cluster per country""")
        st.write(df2.head())
        st.write("""_______________________""")
        st.write("""Below is the Silhouette Visualizer and the corresponding score""")
        st.pyplot(fig_ss)
        st.markdown('Silhouette Score: %.3f' % score_df2)
        st.write("""_______________________""")
        if st.button('Do you have any questions for me?'):
            st.write('Feel free to contact me via sja38@mail.aub.edu')
        else:
            st.write('Hope you found this useful!')

st.sidebar.write("Created by [sjabbar](www.linkedin.com/in/sara-jabbar)")
