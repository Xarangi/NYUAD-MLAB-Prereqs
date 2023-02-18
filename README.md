# NYUAD MLAB PreReqs Bootcamp

## **Linear Algebra**
[Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (1hr 30 min at 2x speed) by 3blue1brown is a gentle video introduction to linear algebra, focusing on building geometric understanding. After this, you should be able to clearly explain the following concepts:
* Matrix-vector multiplication (numerically and geometrically)
* What is and isn’t a linear transformation
* Composition of matrices
* Rank of a matrix
* Basis vectors
* Unit vectors

Videos to watch during bootcamp:
* 2, 3, 4, 9, break (15 min), 10, 13, 14

## **[Pytorch 101](https://colab.research.google.com/drive/1ysF3Q0BpCRmEnl9-X0OH-WMy4KTxx-fj?usp=sharing)**
An introduction to PyTorch, and its functions compared to Numerical Python (Numpy).

## **[Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** by 3blue1brown
Gentle video introduction to neural networks, assuming very little background. The core thing you should know is that an MLP consists of a series of matrix multiplications and (elementwise) non-linearities.

## **Pytorch: Official NN ****[Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)** Tutorial

## **(Extra: Implement an MNIST Classifier)**
A seven step beginner introduction to PyTorch. After this, you should be able to clearly explain:
* At a high level, what is a torch.Tensor?
* What does the requires_grad property do?
* What do we gain by making a class a subclass of nn.Module?
* What is a nn.Parameter? When should you use one?
* When you call backward(), where are your gradients stored?
* What is a loss function? In general, what does it take for arguments, and what does it return?
* What does an optimization algorithm do?
* What is a hyperparameter, and how does it differ from a regular parameter?
* What are some examples of hyperparameters?

You might also like to read [What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html).

**[Max’s guide to einsums/TNDs](https://witty-mirror-0a0.notion.site/Einsum-and-Tensor-Network-Diagrams-cdaed015c6e0440b956d2c208cfbcae5)**
* Matrix multiplication can be generalized to higher-dimensional arrays of numbers. These operations come up frequently in deep learning.
* For this bootcamp, we will focus only on einsums.

## **[Einops Library](https://einops.rocks/1-einops-basics/)**
We will be using the rearrange, reduce, and repeat functions from this library during the course.

We will be providing practice exercises before the course (linked further down)  that you can do to get familiar with these functions.

## **[Final Pre-req Exercises](https://github.com/Kiv/mlab2_pre_exercises/blob/master/w0d1_instructions.md)**
These exercises should be done when you think you’re familiar with all of the above content.

* These exercises will give you some practice working with PyTorch and VSCode.
* If you add lines consisting of #%% to your Python files, you can run them like a Jupyter notebook, i.e. one chunk at a time, by clicking inside the chunk you’d like to run and pressing Shift + Enter.
* Consult the PyTorch and Einops documentation as needed.
* If you find something confusing, ask questions in the #learning channel of the Slack. Please ask questions, and please answer other peoples’ questions if you know the answer.


# **Optional Reading**
These resources are all useful and relevant things to know and will allow you to understand the material of MLAB after the Prerequisites Bootcamp more deeply and/or tackle a more advanced AI Safety Labs project.

## **[100 NumPy Exercises](https://github.com/rougier/numpy-100)**
After you’ve done the pre-course exercises, if you want more practice with this sort of thing, try these. Some of these are a lot more interesting than others - pick out some that sound fun and challenging and try solving them in PyTorch.

## **[What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)** by Jeremy Howard
You’ll be implementing a lot of functionality of torch.nn and torch.optim yourself during the course. This is a good introduction to what functionality is in these packages. If you don’t learn this now, you can pick it up during the course, though perhaps less deeply.

## **[NLP Demystified](https://www.nlpdemystified.org/)** by Nitin Punjabi
An introduction to natural language processing assuming zero ML background.
* If you’ve never done NLP before, it’s worth skimming to get a general idea of the field.
* If you’ve never built a basic feedforward neural network (MLP) from scratch, the section “Neural Networks I” has a good exercise on this.

## **[Visualizing Representations: Deep Learning and Human Beings](https://colah.github.io/posts/2015-01-Visualizing-Representations/)** by Chris Olah
Builds intuition with nice pictures about what deep networks are doing inside.

## **[Pro Git](https://git-scm.com/book/en/v2)** by Scott Chacon and Ben Straub
Goes into Git in much more detail. After the basics, knowing how to stash and rebase are useful skills.

## **[Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)** by OpenAI
Introduction to using deep learning for reinforcement learning. MLAB will assume zero RL experience, but having some understanding already means you’ll have an easier time and can tackle more advanced versions of things in the course.

## **[Introduction to Reinforcement Learning](https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)** with David Silver
This video course is fairly old (2015) and the state of the art has come a long way, but this is still useful to cover the basics. I would recommend Spinning Up in Deep RL over this unless you learn better from video lectures than reading.

## **[The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)** by Petersen and Pedersen
A densely packed reference of facts and identities about matrices. Definitely not intended to teach topics, but a good place to look up something you need. It’s worth memorizing identities 1-6, 11-16, and 18-23 from Page 6.

## **[Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)** by Chris Olah et al
A very fun article on interpretability in neural networks trained for computer vision.

## **[Why Momentum Really Works](https://distill.pub/2017/momentum/)** by Gabriel Goh
Variations of gradient descent that use momentum are extremely common. We’ll teach the basics of momentum in the course, but if you want a richer and deeper understanding then this is a good article to read and reread.

## **[The Matrix Calculus You Need for Deep Learning](https://explained.ai/matrix-calculus/)** by Terence Parr and Jeremy Howard
Takes you from knowing introductory calculus to calculus on matrices. We will teach everything you need from this during MLAB, but you can read ahead if you like.

## **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)** by Anthropic
Analyses transformers starting with the simplest toy models and working up. A heavy read but very good for building intuition about what transformers can do.

## **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)** by Anthropic
Describes and analyses “induction heads”, an important circuit learned by transformers.

## **[Python Type Checking](https://realpython.com/python-type-checking/)** by Geir Arne Hjelle
Python has optional static typing, which we’ll use at times during MLAB. You can figure this out as you go in the course, but if you like having a more comprehensive guide this is a good one. Note that we’re using the built-in type checker of VSCode called Pylance, but it works very similar to MyPy from the article

## **[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)** by Michael Neilson
