# Answer for Questions

1. Logistic Regression should be used for classification tasks.Especially if you need to predict not only labels but propabilities too.
Due to the fact that the logistic regression is very simple, it can be easily interpreted with the help of weights of features. 

2. Linear models are very simple and we have only D parameters. Where D - number of features. However, we can tune some hyperparamaters:
    - Parameter alpha for linear regression(Lasso, Ridge, ElasticNet);(The most important)
    - Parameter C(C = 1/alpha) for logistic regression(SVC too);(The most important)
    - Step of gradient descent;
    - Solver(normal equation, gradient descent, sophisticated gradient descent techniques);

3. A high C value means a more complex model and vice versa. The bigger C means less regularization.

4. For the "heart_*.csv" dataset the most important are:
   Train metrics

    - Accuracy - 0.8677685950413223
    - Recall - 0.9090909090909091
    - Precision - 0.8571428571428571
    - F1 score - 0.8823529411764706

   Test metrics

    - Accuracy - 0.8524590163934426
    - Recall - 0.9090909090909091
    - Precision - 0.8333333333333334
    - F1 score - 0.8695652173913043


   For the "insurance_*.csv" the most important are:
   Train metrics

    - MSE - 37063239.55155356
    - RMSE - 6087.9585701246
    - MAN - 4204.678413334013
   
   Test metrics
	
    - MSE - 33754282.34129212
    - RMSE - 5809.843572876306
    - MAN - 4212.132635671336

5. Metrics for Heart Disease UCI:

      Feature importances
    - cp_0 -0.522251962442931
    - ca_0 0.49286657510639675
    - chol -0.44922833064806483
    - thal_0 -0.4491567648534237
    - cp_2 0.42472471491461783

6. Metrics for Medical Cost Personal:

      Feature importances
    - cp_0 -0.522251962442931
    - ca_0 0.49286657510639675
    - chol -0.44922833064806483
    - thal_0 -0.4491567648534237
    - cp_2 0.42472471491461783
