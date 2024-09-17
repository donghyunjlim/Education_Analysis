"""
This file implements multiple functions using data frame from the National Center for 
Education Statistics. 
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def compare_bachelors_1980(data):
    # Filtering the year and minimum degree
    filtered = data[(data["Year"] == 1980) & (
            data["Min degree"] == "bachelor's")]
    # Filtering gender
    filter_gender = filtered[(filtered["Sex"] == "M") | (
            filtered["Sex"] == "F")]
    return(filter_gender[["Sex", "Total"]])

def top_2_2000s(data, sex="A"):
    """
    Take in two arguments, dataframe and sex(A is a default sex), and
    return a 2 element series (in format of degrees, mean(total))
    which are the two most commonly earned degrees for that given sex
    between the years of 2000 and 2010(both inclusive). If sex parameter
    not specified, the sex is default(A).
    """
    filtered_year = data[(data["Year"] >= 2000) & (data["Year"] <= 2010)]
    filtered_sex = filtered_year[(filtered_year["Sex"] == sex)]
    common_degree = filtered_sex.groupby("Min degree")["Total"].mean()
    return(common_degree.nlargest(2))


def line_plot_bachelors(data):
    """
    Take in a dataframe and plots a line chart of the total percentages
    of all Sex A type people with minimum bachelor's degree over time.
    """
    filtered_df = data[(data["Sex"] == "A") & (
            data["Min degree"] == "bachelor's")]
    sns.relplot(x="Year", y="Total", kind="line", data=filtered_df)
    plt.xlabel("Year")
    plt.ylabel("Percentage")
    plt.title("Percentage Earning Bachelor's over Time")
    plt.savefig("line_plot_bachelors.png", bbox_inches="tight")


def bar_chart_high_school(data):
    """
    Take in a dataframe and plots a bar chart comparing the total
    percentages of all sex types with minimum high school degrees in the
    year of 2009.
    """
    filtered_df = data[(data["Year"] == 2009) & (
            data["Min degree"] == "high school")]
    sns.catplot(x="Sex", y="Total", kind="bar", data=filtered_df)
    plt.xlabel("Sex")
    plt.ylabel("Percentage")
    plt.title("Percentage Completed High School by Sex")
    plt.savefig("bar_chart_high_school.png", bbox_inches="tight")


def plot_hispanic_min_degree(data):
    """
    Take in a dataframe and dot plots the variation of percentage of
    Hispanic people with degrees between 1990 and 2010 (both inclusive) for
    minimum high school and bachelor's degree.
    """
    year = ((data["Year"] >= 1990) & (data["Year"] <= 2010))
    degree = ((data["Min degree"] == "high school") | (
                data["Min degree"] == "bachelor's"))
    filtered_df = data[year & degree]
    sns.relplot(x="Year", y="Hispanic", hue="Min degree", data=filtered_df)
    plt.xlabel("Year")
    plt.ylabel("Percentage")
    plt.title(
        "Percentage of Hispanic with Min Degree of Highschool or Bachelors")
    plt.savefig('plot_hispanic_min_degree.png', bbox_inches='tight')


def fit_and_predict_degrees(data):
    """
    Take in a dataframe and returns the test mean squared error as
    float.
    """
    # selecting columns
    filtered_df = data[["Year", "Min degree", "Sex", "Total"]]
    # Drop rows that have missing data
    filtered_df = filtered_df.dropna()
    features = filtered_df.loc[:, filtered_df.columns != "Total"]
    features = pd.get_dummies(features)
    labels = filtered_df["Total"]
    # Spliting the data randomly into 80% training and 20% testing
    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    # Since it's numeric, regressor
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()

    model.fit(features, labels)
    test_predictions = model.predict(features_test)

    from sklearn.metrics import mean_squared_error
    test_error = mean_squared_error(
        test_predictions, labels_test)
    return(test_error)



def main():
    data = pd.read_csv('nces-ed-attainment.csv', na_values=['---'])
    compare_bachelors_1980(data)
    top_2_2000s(data)
    line_plot_bachelors(data)
    bar_chart_high_school(data)
    plot_hispanic_min_degree(data)
    fit_and_predict_degrees(data)


if __name__ == '__main__':
    main()