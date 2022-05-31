# Shallow Network Approximation Challenge

In this challenge your goal is to obtain a shallow neural network (preferably of a reasonable size) that approximates the data provided in `./data.npy`.

Training and test sets are loaded with ```(x_train, y_train), (x_test, y_test) = np.load('./data.npy', allow_pickle=True)``` and consist of pairs of numbers `(x,y)` that represent the input and the output respectively.

For clarity's sake, the data is sampled from the function `cos(10*pi*x) * (1 - x**2)` on the interval `[-1,1]`, though it is not immediately relevant to this challenge.

If you manage to obtain a shallow network with a decent performance, e.g. the mse-loss on the test set of about `0.01`, please let me know how you did it!


## Setup
Install the dependancies with `pip install -r requirements.txt`, then run with `python main.py`.

## Discussion
Despite a simple formulation, training a shallow network that approximates the given data is surprisingly difficult.
The reasons for the poor performance provided by conventional methods lies in the parameterization of shallow neural networks and the backpropagation-based optimization.
If you're interested in the underlying justification, reach out to me and I'll tell you what I think (*though I might be wrong*)!

![This is what a conventional approach typically provides](https://user-images.githubusercontent.com/38059493/158447963-f679f9a4-a061-4ea5-825a-661b201f3f97.png)

## References
Here are some papers that discuss this phenomenon in more details:
- [Deep ReLU Networks Have Surprisingly Few Activation Patterns](https://arxiv.org/abs/1906.00904)
- [The gap between theory and practice in function approximation with deep neural networks](https://arxiv.org/abs/2001.07523)
- [Training ReLU networks to high uniform accuracy is intractable](https://arxiv.org/abs/2205.13531)
- [Greedy Shallow Networks: An Approach for Constructing and Training Neural Networks](https://arxiv.org/abs/1905.10409)
