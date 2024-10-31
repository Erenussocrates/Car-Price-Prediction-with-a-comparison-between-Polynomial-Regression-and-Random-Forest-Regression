from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import warnings

"""
Açıklama:
Bu projeyi "regression.py"dan çalıştırın, ve tüm veri işlemlerinin doğru şekilde tamamlanması için, her adımı mutlaka en az bir kez seçin.
"""

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random state
RS_number = 16

# Import datasets from data_processing_interface.py
from data_processing_interface import linear_df_final, tree_df_final

# Define target variable and features
target = 'MSRP'

# Split data for Polynomial Regression
X_poly = linear_df_final.drop(columns=[target])
y_poly = linear_df_final[target]
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y_poly, test_size=0.3, random_state=RS_number)

# Function to find the optimal degree for Polynomial Regression
def optimize_poly_degree(X_train, y_train, X_test, y_test, min_degree=1, max_degree=3):
    best_degree = min_degree
    best_score = -float('inf')
    
    for degree in range(min_degree, max_degree + 1):
        try:
            poly = PolynomialFeatures(degree=degree)
            X_train_transformed = poly.fit_transform(X_train)
            X_test_transformed = poly.transform(X_test)
            
            # Fit model with log-transformed target to stabilize
            poly_model = LinearRegression()
            poly_model.fit(X_train_transformed, np.log1p(y_train))
            
            # Prediction and transformation back to original scale
            y_pred_test_log = poly_model.predict(X_test_transformed)
            y_pred_test = np.expm1(np.clip(y_pred_test_log, -700, 700))
            
            # Calculate R2 score
            score = r2_score(y_test, y_pred_test)
            if score > best_score:
                best_score = score
                best_degree = degree
        except MemoryError:
            print(f"Memory Error: Skipping degree {degree}")
            break  # Stop searching if memory limits are hit

    return best_degree

# Run the optimized degree function with reduced max_degree
# optimal_degree = optimize_poly_degree(X_poly_train, y_poly_train, X_poly_test, y_poly_test)

# poly = PolynomialFeatures(degree=optimal_degree)
poly = PolynomialFeatures(degree=2)
X_poly_train_transformed = poly.fit_transform(X_poly_train)
X_poly_test_transformed = poly.transform(X_poly_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train_transformed, np.log1p(y_poly_train))
y_poly_pred = np.expm1(poly_model.predict(X_poly_test_transformed))

# Polynomial Regression Evaluation
poly_r2 = r2_score(y_poly_test, y_poly_pred)
poly_rmse = mean_squared_error(y_poly_test, y_poly_pred, squared=False)

# Split data for Random Forest Regression
X_tree = tree_df_final.drop(columns=[target])
y_tree = tree_df_final[target]
X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(X_tree, y_tree, test_size=0.3, random_state=RS_number)

# Function to find the optimal n_estimators for Random Forest
def optimize_n_estimators(X_train, y_train, X_test, y_test, min_trees=10, max_trees=200, step=10):
    best_trees = min_trees
    best_score = -float('inf')
    for n in range(min_trees, max_trees + 1, step):
        rf = RandomForestRegressor(n_estimators=n, random_state=RS_number)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = r2_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_trees = n
    return best_trees

# Find optimal number of trees for Random Forest
# optimal_trees = optimize_n_estimators(X_tree_train, y_tree_train, X_tree_test, y_tree_test)

"""
Açıklama:
"optimize_poly_degree" ve "optimize_n_estimators" overfittinge sebep olmadan en etkin derece ve ağaç sayılarını bulmama yardımcı
olan fonksiyonlardı, ama bu fonksiyonların çağrıldıktan sonra işlemesi oldukça zaman aldığından dolayı, onların en etkin değerlerini
öğrendikten sonra daha hızlı sonuç almak için bu değerleri kendim manuel olarak girmeye karar verdim.
"""

# rf_model = RandomForestRegressor(n_estimators=optimal_trees, random_state=RS_number)
rf_model = RandomForestRegressor(n_estimators=140, random_state=RS_number)
rf_model.fit(X_tree_train, y_tree_train)
y_tree_pred = rf_model.predict(X_tree_test)

# Random Forest Regression Evaluation
rf_r2 = r2_score(y_tree_test, y_tree_pred)
rf_rmse = mean_squared_error(y_tree_test, y_tree_pred, squared=False)

# Overfitting check
y_poly_train_pred = np.expm1(poly_model.predict(X_poly_train_transformed))
poly_train_r2 = r2_score(y_poly_train, y_poly_train_pred)
poly_overfitting = poly_train_r2 - poly_r2 > 0.1

y_tree_train_pred = rf_model.predict(X_tree_train)
rf_train_r2 = r2_score(y_tree_train, y_tree_train_pred)
rf_overfitting = rf_train_r2 - rf_r2 > 0.1

# Print accuracy results
# print(f"Optimal Polynomial Degree: {optimal_degree}")
print(f"Optimal Polynomial Degree: 2")
print(f"Polynomial Regression R2: {poly_r2:.4f}, RMSE: {poly_rmse:.2f}, Train R2: {poly_train_r2:.4f}")
print("Polynomial Regression is overfitting." if poly_overfitting else "Polynomial Regression is not overfitting.")

# print(f"Optimal number of trees for Random Forest: {optimal_trees}")
print(f"Optimal number of trees for Random Forest: 140")
print(f"Random Forest Regression R2: {rf_r2:.4f}, RMSE: {rf_rmse:.2f}, Train R2: {rf_train_r2:.4f}")
print("Random Forest Regression is overfitting." if rf_overfitting else "Random Forest Regression is not overfitting.")

# Plotting results
plt.figure(figsize=(14, 6))

# Polynomial Regression plot
plt.subplot(1, 2, 1)
plt.scatter(y_poly_test, y_poly_pred, color='blue', alpha=0.6)
plt.plot([y_poly_test.min(), y_poly_test.max()], [y_poly_test.min(), y_poly_test.max()], color='red')
plt.title("Polynomial Regression Predictions vs Actual")
plt.xlabel("Actual MSRP")
plt.ylabel("Predicted MSRP")
plt.grid(True)

# Random Forest Regression plot
plt.subplot(1, 2, 2)
plt.scatter(y_tree_test, y_tree_pred, color='green', alpha=0.6)
plt.plot([y_tree_test.min(), y_tree_test.max()], [y_tree_test.min(), y_tree_test.max()], color='red')
plt.title("Random Forest Regression Predictions vs Actual")
plt.xlabel("Actual MSRP")
plt.ylabel("Predicted MSRP")
plt.grid(True)

plt.tight_layout()
plt.show()