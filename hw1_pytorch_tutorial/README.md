# Homework 1: Pytorch tutorial
**Due Date: 05.03.26 23:59 CET**
**Needs to be solved individually. Gradescope checks for duplicate code.**

## Setup (uv) + Jupyter

### 1) Create a virtual environment

You may use any package manager. We demonstrate the setup with `uv`, an extremely fast Python package installer and manager: https://github.com/astral-sh/uv.

From the repo root:

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
uv pip install torch torchvision jupyter
```

For an introduction on how to use jupyter notebooks, you may check this ressource: https://docs.jupyter.org/en/latest/

## Getting started

All exercises live in `src/`.
Most of these exercises can be solved in very few/one line of code, you're free to use all of torch unless otherwise specified. Usually functions shouldn't change the dtype or device of a tensor. If you receive a dtype
or device make sure to use it for your returned tensors.

### Exercise 1: Tensor basics

Start with **Exercise 1** here:

- `src/ex1_tensor_basics/ex1.ipynb`

Open the notebook, read the short explanations, and **fill in all `TODO`s** in the provided functions.  
Please **do not change function names or signatures**, since the autograder imports these functions directly.

### Exercise 2: PyTorch core

Continue on with **Exercise 2** here:

- `src/ex2_pytorch_core/ex2.ipynb`

Again: Open the notebook, read the short explanations, and **fill in all `TODO`s** in the provided functions.
Please **do not change function names or signatures**, since the autograder imports these functions directly.

### Exercise 3: Neural networks

Then complete **Exercise 3** here:

- `src/ex3_neural_networks/ex3.ipynb`

Again: Open the notebook, read the short explanations, and **fill in all `TODO`s** in the provided functions.
Please **do not change function names or signatures**, since the autograder imports these functions directly.

**Important**: The MNIST classification exercise in ex3 has the additional deliverable of adding a plot of your training and test loss curves to your PowerPoint/slides that you're presenting in your video submission (more details on the video under **2. Assignment submission**). Do not forget to do that! You're welcome to use any plotting utility of your choice.

### Exercise 4: GLU

Finally complete **Exercise 4** here:

- `src/ex4_glu/ex4.ipynb`

For this exercise you will need to read the GLU paper that we provide in the exercise description in the notebook and we encourage you to have a look at the ViT paper that we also provide in that description.
Then **fill in all `TODO`s** in the provided functions to complete this exercise.
**NOTE**: In this exercise we additionally encourage you to think about reproducibility of your results and whether these results are statistically significant. We want you to touch on these points quickly in the video submission. Again, you're welcome to use any plotting utility of your choice for plotting the results.
Please **do not change function names or signatures**, since the autograder imports these functions directly.

**Important**: The GLU exercise has the additional deliverable of adding your results and discussion to the video submission that you will upload. Please read all instructions carefully to not forget about this.

## Submission

### 1. Gradescope Registration

To submit your work, you must be enrolled in the course on Gradescope.

You should've received an email with gradescope details. Please follow the instructions in the email.

If you didn't get an email for gradescope (e.g. you're late enrolled to this course) write Alexey an email: agavryushin@ethz.ch

### 2. Assignment Submission

You will submit **all deliverables** (Code and Video) to the single assignment named **"Homework 1"**.

1.  Navigate to the course page on Gradescope.
2.  Click on the assignment **"Homework 1"**.
3.  Drag and drop (or select) the following files from your computer:
    - `ex1.ipynb`
    - `ex2.ipynb`
    - `ex3.ipynb`
    - `ex4.ipynb`
    - Your video file (`.mp4`)
4.  Click **Upload**.
5.  **Autograding:** A text box will appear showing the autograder's progress on your code. Wait for it to finish to see your preliminary score for the coding exercises.
6.  **Manual Grading:** The teaching assistants will manually review your video after the deadline. Your final score will be updated to include points for the presentation components. Coding problems and the video each count for 50 of the achievable 100 points for this homework.

### Video submission

- ex3 and ex4 require you to show and discuss your results.
- To do this we want you to submit a short video of no longer than 1min of you talking about your results in ex3 and ex4 walking us through the results that you created.
- To show your results we encourage you to put plots that you created in ex3 and ex4 into a quick slide deck which you then walk us through on the short video. Please note we won't grade whether the submission is pretty or not but only whether your reasoning is correct.
- Please make sure your results are visible while explaining (you may use any software to record your screen).
- Please submit the video in `.mp4` format
- You may focus your time in the video on ex4 and only briefly show that ex3 was completed correctly at the beginning.
- We use the video to additionally grade your understanding of ex4. We look for correct reasoning instead of a clear yes or no answer. Please make sure to find the right arguments for your conclusions.
- If you import utilities for plotting or other things please remove them before submitting to the autograder.
