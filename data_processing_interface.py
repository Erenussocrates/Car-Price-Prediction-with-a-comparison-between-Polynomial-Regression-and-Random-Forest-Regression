import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

"""
Açıklama:
Bu projeyi "regression.py"dan çalıştırın, ve tüm veri işlemlerinin doğru şekilde tamamlanması için, her adımı mutlaka en az bir kez seçin.
"""

# Load dataset
data = 'data.csv'
df = pd.read_csv(data)
tree_df_final = df

"""
Açıklama:
Polynomial Regression ve Random Forest Regression'ı ele alacağız, ve bunların verileri farklı işlemlerden geçecek.
"""

# Variable to store skewed columns
skewed_cols = None

print("Bu projeyi 'regression.py'dan çalıştırın, ve tüm veri işlemlerinin doğru şekilde tamamlanması için, her adımı mutlaka en az bir kez seçin.")

# Basic info
while True:

    """
    Açıklama:
    Başlangıçta henüz işlenmemiş (df) bir dataframe'le başladığımız için bazı ilk işlemler bu dataframe üzerinde gösterilecek, ve dataframe'imimiz
    encode edilince aktif dataframe olarak onu kullanıyor olacağız. Her ihtimale karşı veri kaybını önlemek için eski versiyonunu da elde farklı 
    variable olarak tutmak istedim.
    """

    active_df = encoded_df if 'encoded_df' in locals() else df
    
    user_input = input("1) Datasetin şekli\n2) Datasetin sütunları\n3) Datasetin headi\n4) Datasetin infosu\n5) Null sayısı\n6) Sözel sütunların frekansları\n7) Sözel sütunların unique değerleri\n8) Sütunların pairplotlarına bak\n9) KNN ve Simple Imputation Uygulaması\n10) Log Transform uygulaması\n11) Nadiren görülen kategorileri 'Others' olarak gruplandır\n12) Encoding uygulamaları\n13) Korelasyona bak\n14) Yüksek korelasyonları at\nRegression'a devam etmek için herhangi başka birşey girin\n")
    
    if user_input == "1":
        print("Datasetin şekli: ", active_df.shape)
        
    elif user_input == "2":
        print("Datasetin sütunları: ", active_df.columns)
        
    elif user_input == "3":
        print("Datasetin head'i:\n", active_df.head())
        
    elif user_input == "4":
        print("Info of our dataset:\n")
        active_df.info()  

    elif user_input == "5":
        print("Null sayısı:\n", active_df.isnull().sum())
        print("Toplam null sayısı:", active_df.isnull().sum().sum())
        
    elif user_input == "6":
        print("Sözel sütunların frekansları: ")
        for col in active_df.select_dtypes(include='object').columns:
            print(active_df[col].value_counts())
            
    elif user_input == "7":
        print("\nSözel sütunların unique değerleri: \n")
        for col in active_df.select_dtypes(include='object').columns:
            unique_values = active_df[col].dropna().unique()  
            print(f"Column: {col}")
            print("Unique Values:")
            print(unique_values)
            print(f"Total unique values: {len(unique_values)}\n")

    elif user_input == "8":
        print("Generating pairplot for all available columns...")
        sns.pairplot(active_df)
        plt.suptitle("Pairplot of All Available Columns", y=1.02)
        plt.tight_layout(pad=2)
        plt.subplots_adjust(top=0.95)
        plt.show()

        if 'MSRP' in active_df.columns:
            print("Generating histogram for MSRP column...")
            plt.figure(figsize=(10, 6))
            sns.histplot(active_df['MSRP'], kde=True)
            plt.title("Distribution of MSRP")
            plt.xlabel("MSRP")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print("MSRP column not found in active_df.")

    elif user_input == "10":

        """
        Açıklama: 
        Burada, yapacağımız regression modellerinden biri, lineer olan Polynomial Regression olduğu için,
        nümerik feature'ların normal bir dağılımda olması daha iyi sonuç almamız için iyi olur. Ama
        featurelarımız doğal olarak normal bir dağılımda bulunmadıkları için, bunu log transformation
        kullanarak elde etmeye çalıştım. Random Forest Regression için dağılımların büyük bir önemi
        olmadığı için bu ayarlamayı yalnızca Polynomial modelde kullanacağım dataframe için yaptım.
        Bu log transformation'ı aynı zamanda target olan ve normal bir dağılımda olmayan MSRP için de geçerli.
        """

        # Apply log transformation only to non-normally distributed numerical columns
        print("Applying log transformation to non-normally distributed numerical columns...")

        # Define threshold for skewness above which we apply log transformation, best between 0.5 and 1.0
        skew_threshold = 0.5

        # Identify numerical columns in active_df
        numerical_cols = active_df.select_dtypes(include=['float64', 'int64']).columns

        # Find skewed columns and apply log transformation
        skewed_cols = [col for col in numerical_cols if abs(active_df[col].skew()) > skew_threshold]

        if skewed_cols:
            active_df[skewed_cols] = active_df[skewed_cols].apply(lambda x: np.log1p(x))
            print(f"Log transformation applied to the following skewed columns: {skewed_cols}")
        else:
            print("No numerical columns found with skewness above the threshold.")

    elif user_input == "11":

        """
        Açıklama:
        Lineer bir model olan Polynomial Regression'ı kullanırken, özellikle kategorik verilerde olağandışı
        nadirlikler görülürse (tüm örnek uzayın sadece 1%'ini oluşturuyorsa örneğin), bu nadir verileri toplayıp
        ortak bir 'Others' kategorisi altında buluşturmaya karar verdim, çünkü o nadir veriler bırakılırsa, 
        lineer modellerde bunun aşırı model sapmalarına yol açabileceğini öğrendim.
        """

        # Group rare categories in categorical columns
        print("Grouping rare categories under 'Others'...")
        threshold = 0.01  # Define the threshold for rare categories as 1%
        total_rows = len(active_df)
        
        for col in active_df.select_dtypes(include='object').columns:
            # Calculate the frequency of each category
            category_frequencies = active_df[col].value_counts(normalize=True)
            rare_categories = category_frequencies[category_frequencies < threshold].index
            
            # Replace rare categories with 'Others'
            active_df[col] = active_df[col].apply(lambda x: 'Others' if x in rare_categories else x)
        
        print("Rare categories grouped under 'Others'.")

    elif user_input == "12":

        """
        Açıklama:
        Öncellikle One Hot Encoding'i hem Polynomial hem de Random Forest modeli için denedim, ama sonradan gördüm ki
        One Hot Encoding'i feature sayısını aşırı arttırıyor, ve bu özellikle Polynomial Regression featurelarının korelasyonlarını
        incelemeye çalışırken sorun oluşturuyordu. (Random Forest için korelasyon incelemesinin o kadar önemli olmadığını öğrendiğimden,
        Random Forest için yine One Hot Encoding'i kullandım) 
        """

        binary_encoder = BinaryEncoder()
        one_hot_encoder = OneHotEncoder(sparse_output=False)  # Set sparse_output=False to return a dense array
    
        # Binary Encoding for Polynomial Model
        categorical_df = df.select_dtypes(include='object').drop(columns=['MSRP'], errors='ignore')
        encoded_categorical_df = binary_encoder.fit_transform(categorical_df)
        numerical_df = df.select_dtypes(exclude='object')
        encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)

        # One-Hot Encoding for Tree Model
        tree_categorical_df = tree_df_final.select_dtypes(include='object').drop(columns=['MSRP'], errors='ignore')
        encoded_onehot_array = one_hot_encoder.fit_transform(tree_categorical_df)
        one_hot_columns = one_hot_encoder.get_feature_names_out(tree_categorical_df.columns)
        encoded_onehot_df = pd.DataFrame(encoded_onehot_array, columns=one_hot_columns, index=tree_categorical_df.index)
        tree_numerical_df = tree_df_final.select_dtypes(exclude='object')
    
        # Concatenate numerical and encoded categorical DataFrames, then update `tree_df_final`
        tree_df_final = pd.concat([tree_numerical_df, encoded_onehot_df], axis=1)

        print("Binary Encoded DataFrame for Polynomial Model:\n", encoded_df.head())
        print("One Hot Encoded DataFrame for Tree Model:\n", tree_df_final.head())

    elif user_input == "9":
        knn_imputer = KNNImputer(n_neighbors=5)
        encoded_df_float = active_df.select_dtypes(include='float64')
        encoded_tree_float = tree_df_final.select_dtypes(include='float64')
        
        if encoded_df_float.isnull().values.any():
            encoded_df_float_imputed = knn_imputer.fit_transform(encoded_df_float)
            active_df[encoded_df_float.columns] = encoded_df_float_imputed
            print("KNN Imputation applied to float64 columns with null values for Polynomial Model.")
        if encoded_tree_float.isnull().values.any():
            encoded_tree_float_imputed = knn_imputer.fit_transform(encoded_tree_float)
            tree_df_final[encoded_tree_float.columns] = encoded_tree_float_imputed
            print("KNN Imputation applied to float64 columns with null values for Tree Model.")
        
        simple_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
        encoded_df_object = active_df.select_dtypes(include='object')
        encoded_tree_object = tree_df_final.select_dtypes(include='object')
        
        if encoded_df_object.isnull().values.any():
            encoded_df_object_imputed = simple_imputer.fit_transform(encoded_df_object)
            active_df[encoded_df_object.columns] = encoded_df_object_imputed
            print("Simple Imputer (constant) applied to object columns with null values for Polynomial Model.")
        if encoded_tree_object.isnull().values.any():
            encoded_tree_object_imputed = simple_imputer.fit_transform(encoded_tree_object)
            tree_df_final[encoded_tree_object.columns] = encoded_tree_object_imputed
            print("Simple Imputer (constant) applied to object columns with null values for Tree Model.")
        
        print("Imputed DataFrame head for Polynomial Model:\n", active_df.head())
        print("Imputed DataFrame head for Tree Model:\n", tree_df_final.head())

    elif user_input == "13":

        """
        Açıklama:
        Anlaşılan, isabetli korelasyon incelemesi yapabilmek için log transformation'ı yapılmaması gerekiyor, özelliklerin sadece orijinal
        değerlerinde olmaları gerekiyor, o yüzden sadece buradaki çalışma için bu log transformation'ını revert ediyoruz.
        """

        # Revert log transformations on skewed columns before calculating correlations
        if skewed_cols:
            dummy_df = active_df.copy()
            dummy_df[skewed_cols] = dummy_df[skewed_cols].apply(lambda x: np.expm1(x))  # Reverse log transformation
        
            # Plot correlation heatmap on reverted (original scale) data
            plt.figure(figsize=(18, 10))
            heatmap = sns.heatmap(dummy_df.corr(), vmin=-1, vmax=1, annot=True)
            heatmap.set_title('Correlation Heatmap on Original Scale', fontdict={'fontsize': 12}, pad=12)
            plt.show()
        
            print(dummy_df.corr())
        
            # Print the top 5 highest correlations for each feature
            for col in dummy_df:
                en_yuksek_degerler = abs(dummy_df.corr()[col]).nlargest(n=5)
                print(en_yuksek_degerler)
                for index, value in en_yuksek_degerler.items():
                    if 1 > value >= 0.75:
                        print(index, col, "değişkenleri yüksek korelasyona sahip: ", value)
        
            # Print MSRP's correlation with every other column at the bottom
            print("\n--- MSRP Correlations with All Columns ---")
            msrp_correlations = dummy_df.corr()['MSRP'].drop('MSRP').sort_values(ascending=False)
            for col, value in msrp_correlations.items():
                print(f"MSRP and {col}: Correlation = {value:.2f}")
        else:
            print("No log transformations to revert for correlation calculations.")

    elif user_input == "14":

        """
        Açıklama:
        Polynomial modelinin feature'ları için yaptığım korelasyon incelemesinden sonra, korelasyonları yüksek olan feature'ları kıyaslayarak,
        target'a olan korelasyonlarını, ve diğer feature'lara olan korelasyonlarını baz alarak bir puanlama yaptım ve çıkarılacak feature'ları
        bu şekilde belirledim. Dikkatinizi çekerim ki Random Forest modeli için herhangi bir soyutlama işlemi yapılmasına gerek yoktur.
        """

        # Copy active_df into a new DataFrame for linear modeling
        linear_df_final = active_df.copy()
    
        # Specify columns to drop to prevent overfitting in a linear model
        columns_to_drop = ["Engine Cylinders", "city mpg", "Transmission Type_2"]
    
        # Drop the specified columns from linear_df_final
        linear_df_final.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
        print("New DataFrame for linear model created as 'linear_df_final' without specified columns to prevent overfitting.")
        print("Columns removed:", columns_to_drop)
        print("Shape of linear_df_final:", linear_df_final.shape)
        print("Head of linear_df_final:\n", linear_df_final.head())

    else:
        break