import pandas as pd
import numpy as np
import seaborn as sns  # type: ignore
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore

# Lets start out by adding a dataset
df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
# print(df.head(10)); print("ran success")
df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"]; x = df_filtered["age"]







#   Then we can do a linear regression
def linear_regression(x, y):
    sm.add_constant(x)
    model = sm.OLS(y, x).fit()
#   And then measure how accurate it is
    pred = model.predict(x); residuals = y - pred
    LinearError = np.mean(np.abs(residuals))
    print("lin reg error: ", LinearError)
    return LinearError





# Then we can do a quadratic regression
def quadratic_regression(x, y):
    x_quad = np.power(x, 2)
    X = pd.DataFrame({"age": x, "age_squared": x_quad})
    X = sm.add_constant(X)
    model_quad = sm.OLS(y, X).fit()
# And then measure how accurate it is
    pred_quad = model_quad.predict(X)
    residuals_quad = y - pred_quad
    QuadraticError = np.mean(np.abs(residuals_quad))
    print("quad reg error: ", QuadraticError)
    return QuadraticError




# Then we can do a cubic regression
def cubic_regression(x, y):
    x_cube = np.power(x, 3)
    X = pd.DataFrame({"age": x, "age_cubed": x_cube})
    X = sm.add_constant(X)
    model_cube = sm.OLS(y, X).fit()
# And then measure how accurate it is
    pred_cube = model_cube.predict(X)
    residuals_cube = y - pred_cube
    CubeError = np.mean(np.abs(residuals_cube))
    print("cube reg error: ", CubeError)
    return CubeError



# Then we can do a poly regression
def poly_regression(x, y):
    x_4 = np.power(x, 4); x_5 = np.power(x, 5); x_6 = np.power(x, 6); x_7 = np.power(x, 7)

    X_4 = pd.DataFrame({"age": x, "age_4": x_4}); X_4 = sm.add_constant(X_4); model_4 = sm.OLS(y, X_4).fit()
    pred_4 = model_4.predict(X_4); residuals_4 = y - pred_4; Error_4 = np.mean(np.abs(residuals_4))
    X_5 = pd.DataFrame({"age": x, "age_5": x_5}); X_5 = sm.add_constant(X_5); model_5 = sm.OLS(y, X_5).fit()
    pred_5 = model_5.predict(X_5); residuals_5 = y - pred_5; Error_5 = np.mean(np.abs(residuals_5))
    X_6 = pd.DataFrame({"age": x, "age_6": x_6}); X_6 = sm.add_constant(X_6); model_6 = sm.OLS(y, X_6).fit()
    pred_6 = model_6.predict(X_6); residuals_6 = y - pred_6; Error_6 = np.mean(np.abs(residuals_6))
    X_7 = pd.DataFrame({"age": x, "age_7": x_7}); X_7 = sm.add_constant(X_7); model_7 = sm.OLS(y, X_7).fit()
    pred_7 = model_7.predict(X_7); residuals_7 = y - pred_7; Error_7 = np.mean(np.abs(residuals_7))

    print("Polynomial regression (x^4) error: ", Error_4); print("Polynomial regression (x^5) error: ", Error_5)
    print("Polynomial regression (x^6) error: ", Error_6); print("Polynomial regression (x^7) error: ", Error_7)

    return (Error_4, Error_5, Error_6, Error_7)






# Then we can do an exp regression
def exp_regression(x, y):
    log_y = np.log(abs(y)); X = sm.add_constant(x); model_exp = sm.OLS(log_y, X).fit()
    # And then measure how accurate it is
    log_pred = model_exp.predict(X); pred_exp = np.exp(log_pred)
    residuals_exp = y - pred_exp; ExponentialError = np.mean(np.abs(residuals_exp))
    print("exp regression error: ", ExponentialError)
    return ExponentialError



# Log reg
def logarithmic_regression(x, y):
    log_x = np.log(abs(x)); X = sm.add_constant(log_x); model_log = sm.OLS(y, X).fit()
    # And then measure how accurate it is
    pred_log = model_log.predict(X); residuals_log = y - pred_log
    LogarithmicError = np.mean(np.abs(residuals_log))
    print("log regression error: ", LogarithmicError)
    return LogarithmicError




# & then a sign wave regression
def sin_regression(x, y):
    sin_x = np.sin(x); X = sm.add_constant(sin_x); model_sin = sm.OLS(y, X).fit()
# And then measure how accurate it is
    pred_sin = model_sin.predict(X)
    residuals_sine = y - pred_sin
    SinError = np.mean(np.abs(residuals_sine))
    print("sine regression error: ", SinError)
    return SinError



# Logistic
def logistic_regression(x, y):
    y_transformed = (y - np.min(y)) / (np.max(y) - np.min(y))
    X = sm.add_constant(x)
    model_logistic = sm.Logit(y_transformed, X).fit()
    # how accurate
    pred_logistic = model_logistic.predict(X)
    residuals_logistic = y_transformed - pred_logistic
    LogisticError = np.mean(np.abs(residuals_logistic))
    print("logistic regression error: ", LogisticError)
    return LogisticError




# Loess
def loess_regression(x, y, frac=0.3):
    loess_model = sm.nonparametric.lowess(y, x, frac=frac)
    # accuracy
    pred_loess = loess_model[:, 1]; residuals_loess = y - pred_loess
    LoessError = np.mean(np.abs(residuals_loess))
    print("loess regression error: ", LoessError)
    return LoessError



# Then compare how accurate all of the above is

def main(x,y):
    error_list = []

    linear_error = linear_regression(x, y); error_list.append(("Linear", linear_error))
    quadratic_error = quadratic_regression(x, y); error_list.append(("Quadratic", quadratic_error))
    cubic_error = cubic_regression(x, y); error_list.append(("Cubic", cubic_error))
    polynomial_errors = poly_regression(x, y)
    for i, poly_error in enumerate(polynomial_errors, start=4):
        error_list.append((f"Polynomial (x^{i})", poly_error))
    exp_error = exp_regression(x, y); error_list.append(("Exponential", exp_error))
    logarithmic_error = logarithmic_regression(x, y); error_list.append(("Logarithmic", logarithmic_error))
    sine_error = sin_regression(x, y); error_list.append(("Sine", sine_error))
    # logistic_error = logistic_regression(x, y); error_list.append(("Logistic Regression", logistic_error))
    loess_error = loess_regression(x, y); error_list.append(("LOESS", loess_error))

    # Print all errors
    print("\n--- Regression Errors ---")
    for method_name, error in error_list:
        print(f"{method_name} Error: {error}")

    # & designate the most accurate
    min_error_method = min(error_list, key=lambda x: x[1])
    print(f"\nThe method with the smallest error is: {min_error_method[0]} Regression with an error of {min_error_method[1]}")
    print(f"\nTherefore, the function is likely a {min_error_method[0]} function")


main(x,y)


# print(linear_regression(x, y))
