# Customer-Segmentation
# Customer Personality Analysis and Segmentation

This project uses customer demographic and purchasing behavior data to perform **customer segmentation** using unsupervised machine learning. The goal is to group customers into meaningful segments based on income, age, education, spending behavior, purchase activity, promotion response, and family status.

The analysis is completed in a Jupyter Notebook:

* `Customer_Personality_Analysis.ipynb`

## Project Overview

Customer segmentation helps businesses better understand their customers and design more effective marketing strategies. In this project, I use the **Customer Personality Analysis** dataset from Kaggle to clean and transform customer-level data, explore customer behavior, reduce dimensionality using PCA, and apply KMeans clustering to identify distinct customer groups.

Dataset source:

* [Customer Personality Analysis - Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

The dataset contains information for approximately 2,240 customers, including demographics, household information, purchase history, campaign responses, and website activity.

## Objectives

The main objectives of this project are to:

* Clean and prepare customer demographic and shopping data
* Engineer meaningful customer-level features
* Explore distributions and relationships among customer attributes
* Scale the data for clustering
* Use PCA to reduce dimensionality
* Apply KMeans clustering to segment customers
* Evaluate clustering performance using the elbow method and silhouette score
* Interpret customer segments and provide business recommendations

## Methods Used

This project includes the following main steps:

### 1. Data Loading

The dataset is loaded from a tab-separated CSV file:

```python
df = pd.read_csv('marketing_campaign.csv', sep='\t')
```

### 2. Missing Value Handling

The `Income` column contains missing values. Since income is an important factor in customer purchasing behavior, rows with missing income values are removed.

### 3. Data Cleaning

Several irrelevant or constant columns are removed:

* `Z_Revenue`
* `Z_CostContact`
* `ID`

These columns do not provide useful information for clustering.

### 4. Feature Engineering

New features are created to better represent customer behavior and demographics:

| New Feature      | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `age`            | Customer age based on birth year                               |
| `cust_period`    | Number of days the customer has been shopping with the company |
| `has_partner`    | Whether the customer is married/living with a partner          |
| `children`       | Total number of children or teenagers in the household         |
| `education`      | Encoded education level                                        |
| `expenditures`   | Total spending across product categories                       |
| `accepted_offer` | Total number of accepted marketing campaign offers             |
| `num_purchases`  | Total number of purchases across channels                      |

### 5. Outlier Removal

Outliers are removed from:

* `age`, where values above 100 are excluded
* `income`, where extremely high income values are excluded

This helps prevent unrealistic values from distorting the clustering results.

### 6. Exploratory Data Analysis

The notebook includes visual exploration of variables such as:

* Income distribution
* Age distribution
* Education levels
* Boxplots for detecting outliers
* Correlation between features

### 7. Scaling

The cleaned features are scaled using `StandardScaler` before clustering. Scaling is important because KMeans is distance-based and can be affected by differences in feature ranges.

### 8. Determining the Number of Clusters

The project uses:

* Manual elbow method
* `KElbowVisualizer`
* Silhouette score analysis

The initial elbow method suggested five clusters. After applying PCA and evaluating the silhouette scores and business interpretability, four clusters were selected.

### 9. Dimensionality Reduction with PCA

Principal Component Analysis is used to reduce the scaled dataset into three components:

```python
pca = PCA(n_components=3)
```

These components are then used for clustering and 3D visualization.

### 10. KMeans Clustering

KMeans clustering is applied to the PCA-transformed data:

```python
km = KMeans(n_clusters=4)
clusters = km.fit_predict(reduced_df)
```

The final model separates customers into four meaningful groups.

## Technologies Used

* Python
* Jupyter Notebook
* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* Yellowbrick
* SciPy

## Final Features Used

The final cleaned dataset includes the following features:

| Feature          | Description                                       |
| ---------------- | ------------------------------------------------- |
| `income`         | Customer yearly household income                  |
| `recency`        | Number of days since the customer’s last purchase |
| `numwebvisits`   | Number of website visits in the last month        |
| `age`            | Customer age                                      |
| `cust_period`    | Number of days since enrollment with the company  |
| `has_partner`    | Whether the customer lives with a partner         |
| `children`       | Number of children or teenagers in the household  |
| `education`      | Encoded education level                           |
| `expenditures`   | Total amount spent in the last two years          |
| `accepted_offer` | Number of accepted campaign offers                |
| `num_purchases`  | Total number of purchases                         |

## Cluster Summary

The final model identifies four customer segments.

### Cluster 0: Low-Income Younger Customers

This is the largest customer segment. Customers in this group have the lowest income and education levels. They are the youngest group and are likely to live with a partner, with one or no children. They spend the least and show the lowest interest in promotions, but they visit the website more frequently than other groups.

**Business recommendation:**
Target this group online with affordable products and promotions aimed at younger customers.

### Cluster 1: High-Income Single Customers

This group has the highest income and spends the most overall, although they do not have the highest number of purchases. This suggests that they prefer higher-priced products. They are generally single, have no children, and are the most responsive to promotions.

**Business recommendation:**
Target this segment with promotions for premium, non-family-oriented products.

### Cluster 2: Lower-Income Families

Customers in this segment have the second-lowest income level and are the most likely to live with a partner and have children. They have the highest education level on average but spend less overall and are less responsive to promotions.

**Business recommendation:**
Target this group with discounted family-oriented products and value-based offers.

### Cluster 3: Higher-Income Loyal Families

This group has the second-highest income level and includes the oldest customers. They have been shopping with the company for the longest period of time and have relatively high spending. They are likely to live with a partner and children, respond to promotions, and visit the website more often.

**Business recommendation:**
Target this group online with premium family-oriented products and loyalty-based promotions.

## Key Findings

The analysis identified four major customer segments:

1. Low-income, mostly younger customers
2. High-income single customers
3. Lower-income families
4. Higher-income loyal families

These segments can help a business design more personalized marketing campaigns, improve product targeting, and better understand customer behavior.

## Repository Structure

```text
Customer-Personality-Analysis/
│
├── Customer_Personality_Analysis.ipynb   # Main Jupyter Notebook
├── marketing_campaign.csv                # Dataset file, if included
├── README.md                             # Project documentation
└── LICENSE                               # License information
```

## How to Run the Project

1. Clone this repository:

```bash
git clone https://github.com/betulmesci/Customer-Personality-Analysis.git
```

2. Navigate to the project folder:

```bash
cd Customer-Personality-Analysis
```

3. Install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn yellowbrick scipy
```

4. Open the notebook:

```bash
jupyter notebook Customer_Personality_Analysis.ipynb
```

5. Run the notebook cells from top to bottom.

## Future Improvements

Possible future improvements include:

* Testing additional clustering algorithms such as hierarchical clustering or DBSCAN
* Creating clearer visualizations for each customer segment
* Adding a dashboard to explore customer segments interactively
* Comparing clustering results before and after PCA
* Building a marketing recommendation strategy for each segment
* Testing different feature engineering approaches

## Author

**Betul Mescioglu**

Data Science | Python | Machine Learning | Customer Analytics

## License

This project is licensed under the terms included in the `LICENSE` file.
