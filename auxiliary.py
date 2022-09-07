import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import MCA
from wordcloud import WordCloud, get_single_color_func

def load_data():
    """Load data and merge."""
    # Load
    df_ingred = pd.read_csv("Data/ingredients.csv")
    df_recipes = pd.read_csv("Data/recipes.csv")

    return df_ingred, df_recipes


def prepare_sparse_data_analysis(df_ingred, df_recipes):
    """Prepare"""
    # Drop duplicates
    df_ingred.drop_duplicates(inplace=True)
    df_recipes.drop_duplicates(inplace=True)

    # Create Dummy variables of ingredients
    df_ingred["Dummy"] = 1

    # Drop rare ingredients
    rare_ingredients = df_ingred.groupby(["ingredients"]).count().query("ID_recipe <= 20").index
    df_reduced = df_ingred[~df_ingred["ingredients"].isin(rare_ingredients)]

    # Pivot
    df_pivot = df_reduced.pivot_table(
    index="ID_recipe", columns="ingredients", values="Dummy", fill_value=0
    )

    # Pivot
    df_pivot = df_ingred.pivot_table(
        index="ID_recipe", columns="ingredients", values="Dummy", fill_value=0
    )

    # Merge
    res = pd.merge(df_recipes, df_pivot, left_on="ID", right_on="ID_recipe")

    # save
    res.to_csv("Data/own_reduced_merged2_with_all_recipes.csv")


def prepare_sparse_data_analysis_only_unimpaired(df_ingred, df_recipes):
    """Prepare"""
    # Drop duplicates
    df_ingred.drop_duplicates(inplace=True)
    df_recipes.drop_duplicates(inplace=True)

    # Create Dummy variables of ingredients
    df_ingred["Dummy"] = 1

    # Drop rare ingredients
    rare_ingredients = df_ingred.groupby(["ingredients"]).count().query("ID_recipe <= 20").index
    recipe_rare_ingredients = df_ingred.set_index("ingredients").loc[rare_ingredients]["ID_recipe"]
    df_reduced = df_ingred[~df_ingred["ID_recipe"].isin(recipe_rare_ingredients)]

    # Pivot
    df_pivot = df_reduced.pivot_table(
    index="ID_recipe", columns="ingredients", values="Dummy", fill_value=0
    )

    # Pivot
    df_pivot = df_ingred.pivot_table(
        index="ID_recipe", columns="ingredients", values="Dummy", fill_value=0
    )

    # Merge
    res = pd.merge(df_recipes, df_pivot, left_on="ID", right_on="ID_recipe")

    # save
    res.to_csv("Data/own_reduced_merged2.csv")


def prepare_MCA_based_analysis(df_ingred, df_recipes):
    df_res = pd.read_csv("Data/own_reduced_merged2.csv")
    mca = MCA(n_components = 2, n_iter = 3, random_state = 101)

    # take all sparse features
    df_categorical = df_res.drop(['Unnamed: 0', 'ID', 'cuisine'], axis=1).dropna().astype("category")

    # fit and transform 
    mca.fit(df_categorical)
    a_mca = mca.transform(df_categorical)

    # append desired columns
    a_mca["ID"] = df_res["ID"]
    a_mca["cuisine"] = df_res["cuisine"]

    # store as csv
    a_mca.to_csv("Data/own_MCA.csv")

def RF_MCA_parameter_grid():
    #Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(10,80,10)]
    # Number of features to consider at every split
    max_features = ["sqrt"]
    # Maximum number of levels in tree
    max_depth = [2,4]
    # Minimum number of samples required to split a node
    min_samples_split = [2,5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1,2]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the parameter grid
    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': [2,5],
        'min_samples_leaf': [1,2],
        'bootstrap': bootstrap
    }
    return param_grid

def load_MCA_data():
    df = pd.read_csv("Data/own_MCA.csv")
    y_data = df["cuisine"]
    X_data = df.drop(["cuisine", "Unnamed: 0", "ID"], axis=1)

    return X_data, y_data

def load_20_threshold_data():
    df = pd.read_csv("Data/own_reduced_merged2_with_all_recipes.csv")
    y_data = df["cuisine"]
    X_data = df.drop(["cuisine", "Unnamed: 0", "ID"], axis=1)

    return X_data, y_data

def load_20_threshold_data_only_unimpaired():
    df = pd.read_csv("Data/own_reduced_merged2.csv")
    y_data = df["cuisine"]
    X_data = df.drop(["cuisine", "Unnamed: 0", "ID"], axis=1)

    return X_data, y_data

def multiple_salts(df_ingred):
    """Multiple salts"""
    # salt
    salt = df_ingred.set_index("ingredients").filter(regex = r"\bsalt$", axis=0).reset_index(level=0)["ingredients"].unique()

    print(f"There {salt.size} versions of 'salt'':\n")
    text_salt = " ".join(salt)

    # Generate a word cloud image
    #color_func1 = get_single_color_func('deepskyblue')
    x, y = np.ogrid[:1000, :1000]

    mask = (x - 500) ** 2 + (y - 500) ** 2 > 400 ** 2
    mask = 255 * mask.astype(int)
    wordcloud = WordCloud(width=1920, height=1080, background_color='white', mask=mask).generate(text_salt)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    ingred, recipes = load_data()

    res11 = np.array_split(res1, 3)

    res11[0]["Dummy"] = 1
    df_dummy = res11[0].pivot_table(
        index="ID_recipe", columns="ingredients", values="Dummy", fill_value=0
    )

    df_dummy.to_csv("Data/own.csv")
