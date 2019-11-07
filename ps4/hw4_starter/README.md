# CS 4803/7643 Deep Learning - Coding Questions for HW4

In this homework, we will implement algorithms for two kinds - (1) dynamic programming and (2) reinforcement learning with Q-Learning for solving Markov Decision Processes (MDPs).

Note that this homework is adapted from the [Stanford CS234: Reinforcement Learning Winter 2019 course](http://web.stanford.edu/class/cs234/index.html).

Download the starter code [here]({{site.baseurl}}/assets/hw4_starter.zip).

## Setup

Python 3.7.X is required for this assignment. Either install it directly or create a virtual environment with conda:

```
conda create -n hw4 python=3.7
```

First, install dependencies in `requirements.txt`

```
pip install -r requirements.txt
```

Then, install PyTorch 1.2 from [pytorch.org](https://pytorch.org) - either the CPU or GPU version depending on what your machine supports.

## Part 1: Dynamic Programming (20 points + 10 bonus points)

Open the jupyter notebook `dynamic_programming/dp.ipynb` and follow the instructions to implement policy iteration (policy evaluation + policy improvement) and value iteration.


## Part 2: Q-Learning and Deep Q-Networks (30 points + 5 bonus points)

Open the jupyter notebook `q_learning/q_learning.ipynb` and follow the instructions to implement parts of the Q-Learning training procedure and two types of functions for Q networks - a linear Q network and a convolutional Q network.

## Submission

**Step 1:** In `dynamic_programming/dp.ipynb`, make sure you run the entire notebook with the `RENDER_ENV` variable set to `False`, before proceeding to the next step.

**Step 2:** Convert your notebook files to PDF
Option 1: Install [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html) and run `bash convert_to_pdf.sh` to generate the PDF files.
Option 2: If option 1 installation doesn't work, run `bash convert_to_pdf.sh --no-pdf` and manually generate the pdf files by opening the each intermediate HTML file (`dynamic_programming/dp.html` and `q_learning/q_learning.html`) in your browser and saving them as PDFs with the name (`dynamic_programming/dp.pdf` and `q_learning/q_learning.pdf`).

**Step 3:** Submit the two PDF files generated in Step 2 to the assignment titled "HW4" in Gradescope. Assign all pages of `dynamic_programming/dp.pdf` to Question 5.1, and all pages of `q_learning/q_learning.pdf` to Question 5.2.

**Step 3.5** IMPORTANT UPDATE: Please replace your `collect_submission.sh` file by re-downloading the starter code from the original [link]({{site.baseurl}}/assets/hw4_starter.zip) as it has been updated as of November 5th. The older version of the script does not include the Q-Learning jupyter notebook which is required for submission.

**Step 4:** Run `bash collect_submission.sh` to generate `hw4.zip`. Submit this zip file to the assignment titled "HW4 Code" in Gradescope.


#### References:

1. [CS234: Reinforcement Learning Winter 2019 course](http://web.stanford.edu/class/cs234/index.html)
2. [Reinforcement Learning: An Introduction - Sutton & Barto](http://incompleteideas.net/book/RLbook2018.pdf)
3. [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
4. [Playing Atari with Deep Reinforcement Learning - Mnih et. al.](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
5. [Deepmind's Nature Paper on Deep Q-Learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
