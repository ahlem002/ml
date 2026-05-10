import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import sys
sys.path.append('..')
from utils import prepare_hr_data, perform_clustering, get_silhouette_scores

st.set_page_config(page_title="Employee Clustering", page_icon="👥", layout="wide")

st.title("👥 Employee Clustering & HR Analytics")
st.markdown("Analyze employee segments using K-Means clustering")

st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.subheader("⚙️ Settings")
    n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
    show_pca = st.checkbox("Show PCA Visualization", value=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Upload HR Data")
    
    uploaded_file = st.file_uploader("Upload HR CSV file", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write(f"✅ Loaded {len(df)} employees, {len(df.columns)} features")
            
            # Show data info
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            
            # Feature selection
            st.subheader("🎯 Select Features for Clustering")
            
            available_features = [
                "Age", "Education", "Department", "JobRole", "JobLevel",
                "MonthlyIncome", "TotalWorkingYears", "TrainingTimesLastYear",
                "WorkLifeBalance", "JobInvolvement", "PerformanceRating",
                "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
                "YearsWithCurrManager"
            ]
            
            if st.button("🚀 Perform Clustering"):
                try:
                    # Prepare data
                    X_scaled, df_selected, df_encoded = prepare_hr_data(df, available_features)
                    
                    # Perform clustering
                    clusters, kmeans, X_pca, pca = perform_clustering(X_scaled, n_clusters)
                    
                    # Store results in session state
                    st.session_state.X_scaled = X_scaled
                    st.session_state.clusters = clusters
                    st.session_state.kmeans = kmeans
                    st.session_state.X_pca = X_pca
                    st.session_state.pca = pca
                    st.session_state.df_with_clusters = df.copy()
                    st.session_state.df_with_clusters['Cluster'] = clusters
                    st.session_state.df_selected = df_selected
                    
                    st.success(f"✅ Clustering complete! {n_clusters} clusters created.")
                
                except Exception as e:
                    st.error(f"❌ Error during clustering: {e}")
                    st.write(str(e))
        
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

with col2:
    st.subheader("📊 Clustering Results")
    
    if "clusters" in st.session_state:
        clusters = st.session_state.clusters
        df_clusters = st.session_state.df_with_clusters
        
        # Summary statistics
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("Total Employees", len(clusters))
        
        with col_stat2:
            st.metric("Number of Clusters", len(np.unique(clusters)))
        
        with col_stat3:
            unique, counts = np.unique(clusters, return_counts=True)
            avg_size = int(np.mean(counts))
            st.metric("Avg Cluster Size", avg_size)
        
        # Cluster distribution
        st.markdown("---")
        
        cluster_dist = pd.DataFrame({
            'Cluster': np.unique(clusters),
            'Employee Count': [np.sum(clusters == c) for c in np.unique(clusters)]
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(cluster_dist['Cluster'], cluster_dist['Employee Count'], color='#FF6B6B')
        ax.set_title('Employee Distribution by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Employees')
        st.pyplot(fig)
        
        # Cluster details table
        st.write("**Cluster Summary:**")
        st.dataframe(cluster_dist, use_container_width=True)

# Visualizations
st.markdown("---")
st.subheader("📈 Visualizations")

if "clusters" in st.session_state and show_pca:
    X_pca = st.session_state.X_pca
    clusters = st.session_state.clusters
    
    col_viz1, col_viz2 = st.columns([1, 1])
    
    with col_viz1:
        st.markdown("#### PCA 2D Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))

        if X_pca.shape[1] > 1:
            y_vals = X_pca[:, 1]
            y_label = f"PC2 ({st.session_state.pca.explained_variance_ratio_[1]:.2%})"
        else:
            # Fallback for single-feature inputs where PCA yields only one component.
            y_vals = np.zeros(len(X_pca))
            y_label = "Component 2 (not available)"

        scatter = ax.scatter(X_pca[:, 0], y_vals, c=clusters, cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel(f"PC1 ({st.session_state.pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(y_label)
        ax.set_title("Employee Clusters (PCA)")
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
    with col_viz2:
        st.markdown("#### Cluster Characteristics")
        
        df_clusters = st.session_state.df_with_clusters
        
        # Show cluster centers info
        try:
            numeric_cols = df_clusters.select_dtypes(include=[np.number]).columns.tolist()
            if 'Cluster' in numeric_cols:
                numeric_cols.remove('Cluster')
            
            if numeric_cols:
                selected_col = st.selectbox("Select feature to analyze:", numeric_cols)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                df_clusters.boxplot(column=selected_col, by='Cluster', ax=ax)
                ax.set_title(f"{selected_col} by Cluster")
                ax.set_xlabel("Cluster")
                plt.suptitle('')  # Remove default title
                st.pyplot(fig)
        except Exception as e:
            st.info("Select numeric features for analysis")

# Silhouette Analysis
st.markdown("---")
st.subheader("🔍 Silhouette Analysis")

if "X_scaled" in st.session_state:
    if st.button("Calculate Silhouette Scores"):
        with st.spinner("Calculating..."):
            scores = get_silhouette_scores(st.session_state.X_scaled, max_clusters=10)
            
            cluster_nums = [s[0] for s in scores]
            silhouette_scores = [s[1] for s in scores]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cluster_nums, silhouette_scores, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score vs Number of Clusters")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Find optimal
            optimal_k = cluster_nums[np.argmax(silhouette_scores)]
            st.info(f"💡 Optimal number of clusters: **{optimal_k}** (Silhouette Score: {max(silhouette_scores):.3f})")

# Download results
if "df_with_clusters" in st.session_state:
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    csv = st.session_state.df_with_clusters.to_csv(index=False)
    st.download_button(
        label="Download Clustered Data (CSV)",
        data=csv,
        file_name="employee_clusters.csv",
        mime="text/csv"
    )

st.markdown("---")
st.info("""
**How it works:**
1. Upload HR dataset (CSV/Excel)
2. Select number of clusters
3. System performs K-Means clustering
4. Visualize results with PCA
5. Analyze cluster characteristics
6. Download results for further use

**Typical HR Features:**
- Demographics: Age, Education, Department, Job Role
- Experience: Total years, years in company, years in role
- Engagement: Training, work-life balance, job involvement
- Performance: Rating, promotion history
""")
