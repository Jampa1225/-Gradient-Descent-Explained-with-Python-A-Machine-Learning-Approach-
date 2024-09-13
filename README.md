# Gradient-Descent-Explained-with-Python-A-Machine-Learning-Approach

In this tutorial you can learn how the gradient descent algorithm works and implement it from scratch in python. First we look at what linear regression is, then we define the loss function. We learn how the gradient descent algorithm works and finally we will implement it on a given data set and make predictions.



![1_CjTBNFUEI_IokEOXJ00zKw](https://github.com/user-attachments/assets/16748599-266d-47fb-8a4a-01a5914e44c4)

# Linear Regression

In statistics, linear regression is a linear approach to modelling the relationship between a dependent variable and one or more independent variables. Let X be the independent variable and Y be the dependent variable. We will define a linear relationship between these two variables as follows:

![LR Formula Pic](https://github.com/user-attachments/assets/2871a8dc-001c-4d7c-aa4f-b7d24f64d9e3)


![1_ETn5o9GRaF8ZK6wIHvGrJQ](https://github.com/user-attachments/assets/1a8e58fa-a651-4188-9213-8e6023cb7843)

This is the equation for a line that you studied in high school. m is the slope of the line and c is the y intercept. Today we will use this equation to train our model with a given dataset and predict the value of Y for any given value of X. Our challenge today is to determine the value of m and c, such that the line corresponding to those values is the best fitting line or gives the minimum error.

# Loss Function

The loss is the error in our predicted value of m and c. Our goal is to minimize this error to obtain the most accurate value of m and c.
We will use the Mean Squared Error function to calculate the loss. There are three steps in this function:

1 - Find the difference between the actual y and predicted y value(y = mx + c), for a given x.

2 - Square this difference.

3 - Find the mean of the squares for every value in X.

![Cost-loss-Error Function Formula pic](https://github.com/user-attachments/assets/6cbda75e-27b8-48c5-9c49-04537f681916)

Here yᵢ is the actual value and ȳᵢ is the predicted value. Lets substitute the value of ȳᵢ:


![cost-loss-error Function formula pic 1](https://github.com/user-attachments/assets/c496db31-8e3b-48f1-972f-fe18a27a524f)

So we square the error and find the mean. hence the name Mean Squared Error. Now that we have defined the loss function, lets get into the interesting part — minimizing it and finding m and c.

# The Gradient Descent Algorithm

# Objective

Gradient descent algorithm is an iterative process that takes us to the minimum of a function(barring some caveats). The formula below sums up the entire Gradient Descent algorithm in a single line.


![GD fund Formula Pic](https://github.com/user-attachments/assets/e9bb7120-12c4-4a40-8546-403817a27f33)

But how do we arrive at this formula? Well, It is straightforward and includes some high school maths. Through this article, we shall try to understand and recreate this formula in the context of a Linear Regression model.

# A Machine Learning Model

* Consider a bunch of data points in a 2 D space. Assume that the data is related to the height and weight of a group of students. We are trying to predict some relationship between these quantities to predict the weight of some new students afterward. This is essentially a simple example of a supervised Machine Learning technique.

* Let us now draw an arbitrary line in space that passes through some of these data points. The equation of this straight line would be Y = mX + b where m is the slope and b is its intercept on the Y-axis.

![LR Pic](https://github.com/user-attachments/assets/fd3f0444-d7ea-4e12-a9ec-1fee782542db)

# Predictions

Given a known set of inputs and their corresponding outputs, A machine learning model tries to make some predictions for a new set of inputs.

![Pic 1](https://github.com/user-attachments/assets/ec03481a-4ed1-4672-a582-b7074ac3f649)

The Error would be the difference between the two predictions.

![Pic 2](https://github.com/user-attachments/assets/fe193134-d6d2-427d-9f3a-f11a0bc0cf77)

This relates to the idea of a Cost function or Loss function.

# Cost Function

A Cost Function/Loss Function evaluates the performance of our Machine Learning Algorithm. The Loss function computes the error for a single training example, while the Cost function is the average of the loss functions for all the training examples. Henceforth, I shall be using both the terms interchangeably.

# A Cost function basically tells us ‘ how good’ our model is at making predictions for a given value of m and b.
Let’s say there are a total of ’N’ points in the dataset, and for all those ’N’ data points, we want to minimize the error. So the Cost function would be the total squared error i.e

![Cost Pic 1](https://github.com/user-attachments/assets/be7ad4f2-7495-48ef-a3ea-c1369ad542fd)

Why do we take the squared differences and simply not the absolute differences? Because the squared differences make it easier to derive a regression line. Indeed, to find that line we need to compute the first derivative of the Cost function, and it is much harder to compute the derivative of absolute values than squared values. Also, the squared differences increase the error distance, thus, making the bad predictions more pronounced than the good ones.

# Minimizing the Cost Function

The goal of any Machine Learning Algorithm is to minimize the Cost Function.

This is because a lower error between the actual and predicted values signifies that the algorithm has done an excellent job learning. Since we want the lowest error value, we want those‘ m’ and ‘b’ values that give the smallest possible error.

# How do we minimize any function?

If we look carefully, our Cost function is of the form Y = X² . In a Cartesian coordinate system, this is an equation for a parabola and can be graphically represented as :

![cost pic 2](https://github.com/user-attachments/assets/132a2516-c424-4d86-b001-8ffefaeedc09)

To minimize the function above, we need to find that value of X that produces the lowest value of Y which is the red dot. It is pretty easy to locate the minima here since it is a 2D graph, but this may not always be the case, especially in higher dimensions. For those cases, we need to devise an algorithm to locate the minima, and that algorithm is called Gradient Descent.

# The Gradient Descent Algorithm

Gradient descent is an iterative optimization algorithm to find the minimum of a function. Here that function is our Loss Function.

Understanding Gradient Descent

![GD Mountain Pic](https://github.com/user-attachments/assets/7450590d-db09-4966-b8ba-5b5fbcecc232)

Imagine a valley and a person with no sense of direction who wants to get to the bottom of the valley. He goes down the slope and takes large steps when the slope is steep and small steps when the slope is less steep. He decides his next position based on his current position and stops when he gets to the bottom of the valley which was his goal.

Let’s try applying gradient descent to m and c and approach it step by step:

1 - Initially let m = 0 and c = 0. Let L be our learning rate. This controls how much the value of m changes with each step. L could be a small value like 0.0001 for good accuracy.

2 - Calculate the partial derivative of the loss function with respect to m, and plug in the current values of x, y, m and c in it to obtain the derivative value D.

![Loss Function 1](https://github.com/user-attachments/assets/0a94d1d3-13f4-47e5-9235-f0ac7636b30e)

Dₘ is the value of the partial derivative with respect to m. Similarly lets find the partial derivative with respect to c, Dc :

![Loss Function Derivativation 1](https://github.com/user-attachments/assets/ae185a83-16d9-448b-92d0-875cb5ec5a2a)

3. Now we update the current value of m and c using the following equation:


![New weight GD pic](https://github.com/user-attachments/assets/1f925189-60d7-45b0-888d-b65af6c0651b)

4. We repeat this process until our loss function is a very small value or ideally 0 (which means 0 error or 100% accuracy). The value of m and c that we are left with now will be the optimum values.

Now going back to our analogy, m can be considered the current position of the person. D is equivalent to the steepness of the slope and L can be the speed with which he moves. Now the new value of m that we calculate using the above equation will be his next position, and L×D will be the size of the steps he will take. When the slope is more steep (D is more) he takes longer steps and when it is less steep (D is less), he takes smaller steps. Finally he arrives at the bottom of the valley which corresponds to our loss = 0.
Now with the optimum value of m and c our model is ready to make predictions !

# Implementing the Model
Now let’s convert everything above into code and see our model in action !

![GD imple pic 1](https://github.com/user-attachments/assets/86aa3f4b-c569-430d-ae21-e3e326bf553c)

![GD best fit line pic 1](https://github.com/user-attachments/assets/1a7cff91-29f2-43eb-a2d1-d9ec3d45204b)

Gradient descent is one of the simplest and widely used algorithms in machine learning, mainly because it can be applied to any function to optimize it. Learning it lays the foundation to mastering machine learning.

