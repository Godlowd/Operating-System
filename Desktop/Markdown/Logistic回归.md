# Logistic Regression

h$\theta $~(x) = g($\theta$ ^T^x)
g(z) = $\frac{1}{ 1 + e^{-z}}$

predict  "y = 1" if $\theta $^T $\geq$ 0 , also "y= 0" if $\theta$^T < 0

* decision boundary seperates the region , which is **a property of the hypothesis** ,not **a property of the data set**

* By adding more complex polynomial terms to make the decision boundary more complex

## the optimization objective cost
* hava a training set of **M** training examples 
* each of our example is represented by **feature vector that's n plus 1 dimensional**
* x~0~ = 1 , y $\in $ {0,1}
* h~$\theta $~ =  $\frac{1}{ 1 + e^{-z}}$

### Logistic regression cost function 

$$
J(\theta) = \frac{1}{m}\sum_{i = 1}^{m}Cost(h_\theta(x) ,y)
$$

Notes: y =0 or 1 **always**
$$
Cost(h_\theta(x) ,y ) = 
			\begin{cases} 
			-\log(h_\theta(x)), & \text{if $y$ = 1}\\
			-\log(1-h_\theta(x)), & \text{if $y$ = 0}
			\end{cases}
$$
* A simpler way to write the cost function

$$
Cost(h\theta(x)) = -y\log (h\theta(x)) - (1-y)\log(1-h\theta(x))
$$

## Advanced optimization methods

* A  Cost function J($\theta$) . We want tp minimize the J($\theta$)
* More sophisticated.faster ways to optimize the $\theta$ that can be used instead of gradient descent
1. Conjugate gradient 

2. BFGS

3. L-BFGS

 We first need to provide a function that evaluates the **J($\theta$)**  and **$\frac{\partial J(\theta)}{\partial \theta~j~}$** for a given $\theta$ 
```octave
function [jVal, gradient] = costFunction(theta)	//返回jVal与gradient的函数
jVal = [Code the compute the J(theta)]			//计算给定theta的代价函数的代码
gradient = [Code to compute the derivative of J(\theta)]	//计算给定theta的导数的代码
end
```

Then we can us octave's **fminunc** optimization algorithm along with the **optimset** function that creates an objeccs **contaning the options we want to set the "fminunc"**

```octave
options = optimset('GradObj', 'on','MaxIter',100);
initialTheta = zero(2,1);
	[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

## one-versus-all classification 

y can take some **discrete values**,maybe 1 to 3, 1 to 4 and so on. 

## overfitting and regularization
* a algorithm with high variance 
* don't have enough data to constrain it to get a good hypothesis

### Addressing overfitting 
Options: 
1. **Reduce** number of features
2.  **Regularization**

# Neural Network
* activation function
g(z) = $\frac{1}{1+e^{-z}}$
* **weight**/**parameters $\theta$** is the same thing 
* The first layer is also called **the input layer** ,where we input our features
* The final layer is also called **the output layer** ,that output s the final value .
*  a~i~^(j)^ , **neuron i** or **unit i** in **layer j**  
* $\theta$^(j)^ , matrix of weights controlling function mapping from **layer j** to **layer j + 1 ** 

* L : total number of layers in network
* s~l~: number of units in **layer l**(**not counting bias unit**)



so in this graph, s~1~ = 3, s~2~ = 5,s~4~ = s~L~ =4(because L = 4)

1. Binary Classification
	y are either **0** or **1** .With one output unit 
2. Multi-class Classification(K classes)
	With k output units
## Cost function
**Logistic regression**:
$$
J(\theta) =-\frac{1}{m}[\sum_{i=1}^{m}y^i \log h_\theta (x^i) + (1-y^i)\log (1-h_\theta(x^i))] + \frac{\lambda}{2m}\sum_{j=1}^{n} \theta_j^2
$$
**Neural Network** 
$$
J(\theta) =-\frac{1}{m}[\sum_{i=1}^{m}\sum_{k=1}^{K}y_K^i \log (h_\theta (x^i))_k + (1-y^i)\log (1-h_\theta(x^i))_k] + \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_l=1} \theta_j^2
$$
Notes: (h~\theta~ (x))~i~ = i^th^ output, h(x) is a **K dimensional vector** 

## Algorithm to minimize the cost function(the back propagation algorithm)

Goal: To find a $\theta$ to minimize the cost function

Method: use **gradient descent** or **other advandced algorithms**, we need **code to compute the J($\theta$) and the partial derivate terms** 

$\delta _j^l$  = **error** of node j in layer l

For each output unit (layer L = 4), $\delta$ ^4^ = a~j~ ^4^ - y~j~ 



