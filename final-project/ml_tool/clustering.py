#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


def plot_violin_for_features(data, clusters, feature_columns):
    """
    יצירת גרפי Violin לכל עמודה מספרית בהשוואה בין קלאסטרים.
    :param data: DataFrame - הדאטה
    :param clusters: array - הקלאסטרים שנוצרו
    :param feature_columns: list - שמות העמודות שברצונך להציג בגרף
    """
    for feature in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=clusters, y=data[feature], palette="viridis")
        plt.title(f"Violin Plot for {feature} by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(feature)
        st.pyplot(plt)

def find_optimal_clusters(data, max_clusters=10):
    """
    פונקציה למציאת מספר הקלאסטרים האופטימלי בעזרת Elbow Method ו-Silhouette Score.
    :param data: DataFrame - הדאטה לניתוח
    :param max_clusters: int - המספר המרבי של קלאסטרים לבדיקה
    :return: int - מספר הקלאסטרים האופטימלי
    """
    distortions = []
    silhouette_scores = []
    
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        clusters = kmeans.fit_predict(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, clusters))
    
    # Finding optimal clusters based on highest Silhouette score
    optimal_clusters = np.argmax(silhouette_scores) + 2
    return optimal_clusters

def run_kmeans_clustering(data, optimal_clusters):
    """
    הפעלת קלאסטרינג עם KMeans
    :param data: DataFrame - הדאטה לניתוח
    :param optimal_clusters: int - מספר הקלאסטרים האופטימלי
    :return: KMeans model - המודל המאומן
    """
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters

def run_dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    הפעלת קלאסטרינג עם DBSCAN
    :param data: DataFrame - הדאטה לניתוח
    :param eps: float - המרחק המקסימלי בין נקודות
    :param min_samples: int - מספר מינימלי של נקודות בקלאסטר
    :return: DBSCAN model - המודל המאומן
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return dbscan, clusters

def find_patterns_and_importance(data, max_clusters=10, clustering_algorithm='KMeans'):
    """
    פונקציה שמבצעת קלאסטרינג ומציגה Feature Importance וניתוחים נוספים
    :param data: DataFrame - הדאטה לניתוח
    :param max_clusters: int - מספר הקלאסטרים המרבי לבדיקה
    :param clustering_algorithm: string - אלגוריתם קלאסטרינג לבחירה ('KMeans' או 'DBSCAN')
    """
    st.write("### Step 1: Selecting Numeric Columns for Clustering")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # מציאת מספר הקלאסטרים האופטימלי
    optimal_clusters = find_optimal_clusters(numeric_data, max_clusters)
    st.write(f"Optimal number of clusters found: {optimal_clusters}")

    # ביצוע קלאסטרינג על פי האלגוריתם שנבחר
    if clustering_algorithm == 'KMeans':
        model, clusters = run_kmeans_clustering(numeric_data, optimal_clusters)
        st.write(f"Clusters assigned by KMeans: {np.unique(clusters)}")
    elif clustering_algorithm == 'DBSCAN':
        model, clusters = run_dbscan_clustering(numeric_data)
        st.write(f"Clusters assigned by DBSCAN: {np.unique(clusters)}")
    
    # הוספת הקלאסטרים לדאטה
    data['Cluster'] = clusters
    
    # ניתוח Feature Importance בעזרת Random Forest
    st.write("### Step 2: Analyzing Feature Importance Using Random Forest")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(numeric_data, clusters)
    importances = rf.feature_importances_
    
    # הצגת Feature Importance
    feature_importance_df = pd.DataFrame({
        'Feature': numeric_data.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.write("We trained a Random Forest model to assess which features were most important in determining the clusters.")
    st.dataframe(feature_importance_df)

    # הצגת Feature Importance בגרף
    st.write("### Step 3: Feature Importance Visualization")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette="viridis")
    plt.title("Feature Importance Based on Clustering")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    st.pyplot(plt)

    # 4. ויזואליזציה של הקלאסטרים באמצעות PCA
    st.write("### Step 4: Visualizing Clusters Using PCA")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(numeric_data)
    st.write("PCA was used to reduce the dimensionality to 2 components for visualization.")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette="viridis", s=100)
    plt.title("Clustering Visualization (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    st.pyplot(plt)
    
    # 5. יצירת גרפים Violin עבור כל התכונות
    st.write("### Step 5: Violin Plot for Features by Cluster")
    plot_violin_for_features(numeric_data, clusters, numeric_data.columns)
    
    # 6. הערכת איכות הקלאסטרים
    st.write("### Step 6: Evaluating Clustering Quality")
    silhouette = silhouette_score(numeric_data, clusters)
    davies_bouldin = davies_bouldin_score(numeric_data, clusters)
    st.write(f"Silhouette Score: {silhouette}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin}")
    
    # 7. הצגת תובנות על קשרים בין פיצ'רים
    st.write("### Step 7: Correlation Heatmap of Features")
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Features")
    st.pyplot(plt)

    # 8. הצגת תובנות על קשרים בין פיצ'רים
    st.write("### Step 8: Insights from Feature Correlations")
    st.write("The heatmap above shows the correlation between different features. Highly correlated features can help explain the underlying relationships in the data.")
    
    # 9. הצגת תובנות לפי קלאסטרים
    st.write("### Step 9: Insights for Each Cluster")
    for cluster_num in np.unique(clusters):
        st.write(f"Insights for Cluster {cluster_num}:")
        cluster_data = data[data['Cluster'] == cluster_num]
        cluster_summary = cluster_data.describe()
        st.write(f"Cluster {cluster_num} Summary:")
        st.write(cluster_summary)




