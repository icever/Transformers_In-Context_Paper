# What Can Transformers Learn In-Context? A Case Study of Simple Function Classes

## Practice


<details>
<summary>Q1</summary>

![image](./gpt_in_context_01.png)    

<details>
<summary>Q1_Answer</summary>

![image](./gpt_in_context_01_ans.png)    
</details>    

</details>    

<details>
<summary>Q2</summary>  

![image](./gpt_in_context_02.png)    

<details>
<summary>Q2_Answer</summary>

![image](./gpt_in_context_02_ans.png)    
</details>    

</details>    

## In-Context Learning

- **Input-Output Pairs within Prompt**:

    The core of in-context learning involves leveraging input-output pairs provided within the prompt. These pairs exemplify the task at hand, allowing the model to infer the underlying pattern or function to apply to a new input query.

- **Without Explicit Training, Output at Inference Time**:

    One of the distinctive features of in-context learning is its ability to adapt and generate outputs without any modifications to the model's parameters during inference. This means the model relies solely on the information provided in the prompt to perform the task, showing a form of learning and generalization that occurs without the traditional training process.

## Problems 

- Can we train a model to in-context learn a certain function class?

- What extent do Transformers develop the ability to learn new tasks from in-context examples alone?

## Formal Approach for In-Context Learning

$$ \mathbb{E}_P [ \ell ( M(P), f(x_{\text{query}}) ) ] ≤ \epsilon $$ 

- **Function Class**: 

    The paper defines a model's capability to in-context learn a function class $F$ as its ability to predict outputs for new queries based on a series of in-context examples. This approach is crucial for assessing a model's adaptability and learning efficiency.

    Using 20-dimentional inputs $ \mathbf{x} $, we will discuss the 4 function  classes as follows.

    - **Linear Function**

        - $ f(\mathbf{x}) = w_1x_1 + w_2x_2 + ... + w_{20}x_{20} + b $ (21 parameters)
    - **3-Sparse Linear Function**

        - $ f(\mathbf{x}) = w_2x_2 + w_5x_5 + w_{20}x_{20} $ (3 parameters)

    - **Two-Layer ReLU Neural Networks with 100 Hidden Units**

        - $ f(\mathbf{x}) = \mathbf{W}_2^\top(\text{ReLU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)) + b_2 $ (Assuming a single output unit, 2201 parameters - 20*100 + 100 + 100 +1)

    - **Decision Tree of Depth 4**
        - $ f(\mathbf{x}) $ is a series of if-else conditions upto 15 decisions.
        - Example:
            - If $ x_1 > \text{threshold}_1 $:
                - If $ x_2 > \text{threshold}_2 $: Return Class A
                - Else: Return Class B
            - Else:
                - If $ x_2 > \text{threshold}_3 $: Return Class C
                - Else: Return Class D    

- **Prompt Structure and Learning Process**: 

    A prompt consists of input-output pairs derived from functions in $F$ and a new query input. These inputs are independently drawn from a distribution over inputs ($D_X$), and the function is chosen from a distribution over functions ($D_F$). This setup simulates how models encounter and process new information.

    - Input and Function Distribution ($D_X$ and $D_F$) follows multivariate isotropic Gaussian distribution $ N(0, I_d) $.
        - $ N $ denotes the Gaussian or normal distribution.
        - $ 0 $ represents the mean vector, which, in this case, is a vector of zeros. The length of this vector matches the number of dimensions $ d $.
        - $ I_d $ is the identity matrix of size $ d \times d $. An identity matrix is a square matrix with ones on the main diagonal and zeros elsewhere.

- **Formal Criteria for In-Context Learning**: 

    The paper sets a formal criterion for successful in-context learning, measured by the model's ability to predict the output for a new query input with an average error below a certain threshold ($\epsilon$). This metric allows for a quantifiable evaluation of a model's in-context learning performance.

## Training

![image](./training.png)

The training process described for the Transformer model to enable it to perform in-context learning can be explained through the following points:

- **Training Regime**:
    
    GPT-2 family architectures
    ![image](./gpt2_models.png)

    In this research, the GPT-2 architecture is used as a starting point for training from scratch, meaning that no pre-training on language data or fine-tuning of pre-existing models is done. The focus is solely on the ability to learn function mappings in-context.

- **Training Objective**: 

    The training objective is to minimize the expected loss over all such prompt prefixes, using a chosen loss function, typically squared error in this case. The expected loss is calculated as the average loss over the prompt prefixes for a batch of prompts.

- **Gradient Updates**: 

    At each training step, the model parameters $ \theta $ are updated via gradient descent to reduce the loss. Batches of random prompts are used for each update, with the Adam optimizer often chosen for its efficiency and effectiveness.

- **Hyperparameters**:
    
    The model is trained for a significant number of steps (e.g., 500,000) with a standard batch size (e.g., 64), and uses a set learning rate (e.g., $ 10^{-4} $) for all function classes and models.

- **Process**:
    1. **Sampling Functions and Inputs**: The process begins by sampling random functions from a function class $ F $, according to a distribution $ D_F $. For each sampled function $ f $, a sequence of inputs $ x_1, ..., x_{k+1} $ is drawn from an input distribution $ D_X $. These inputs are then used to generate the corresponding outputs using the sampled function to create training prompts.

    2. **Constructing Prompts**: A prompt $ P $ is constructed using the input-output pairs $ (x_1, f(x_1)), ..., (x_{k+1}, f(x_{k+1})) $. For linear functions, for instance, the inputs are drawn from an isotropic Gaussian distribution $ N(0, I_d) $, and the function is defined using a weight vector $ w $ also drawn from $ N(0, I_d) $ such that $ f(x) = w^\top x $.

    3. **In-Context Prediction Training**: The Transformer is trained to predict the output for a given input $ x_i $ based on a set of preceding in-context examples. Specifically, for each input $ x_i $ within a prompt, the model uses the previous $ i $ input-output pairs as context to predict the output of $ x_{i+1} $.

## Pseudocode for In-Context Learning
```
Algorithm: Validate In-Context Learning Performance of Transformer

/* Transformer, a model for in-context learning, evaluation phase */

Input: 
  M, a trained Transformer model
  S, a set of sampled prompts P with corresponding true outputs
  L, a loss function
  ε, an error threshold
Output: 
  Validation that expected loss is within the threshold ε

Procedure:
1: Initialize total_loss to 0
2: for each prompt-output pair (P, f(x_query)) in S do
3:     x_query ← the query input from P
4:     M_output ← M(P)                  // Get model's prediction for x_query
5:     loss ← L(M_output, f(x_query))   // Compute loss between prediction and true output
6:     total_loss ← total_loss + loss
7: end for
8: average_loss ← total_loss / |S|      // Calculate the average loss over all prompts
9: if average_loss ≤ ε then
10:    return "Model meets in-context learning performance criteria."
11: else
12:    return "Model does not meet in-context learning performance criteria."
13: end if

/* End Algorithm */
```


```
Algorithm 1: Train In-Context Learning Transformer

/* Training a Transformer model for in-context learning */

Input: 
  F, a function class (e.g., linear functions)
  D_F, a distribution over functions in F
  D_X, a distribution over inputs
  k, the number of in-context examples
  LossFunction, the loss function to use (e.g., squared error)

Hyperparameters: 
  η, learning rate
  batch_size, number of prompts per training batch
  total_steps, total number of training iterations

Parameters: 
  θ, model parameters to be learned

Output: 
  θ*, optimized model parameters after training

Procedure:
1: Initialize model parameters θ randomly
2: for step in 1 to total_steps do
3:     Initialize batch_loss to 0
4:     for prompt_index in 1 to batch_size do
5:         f ← sample function from D_F
6:         X ← sample k+1 inputs from D_X
7:         Y ← evaluate f on inputs X to get k+1 outputs
8:         P ← construct prompt (X, Y) with k in-context examples
9:         for i in 1 to k do
10:            X_i, Y_i ← first i examples from P
11:            y_hat ← M(X_i, θ)  // Model's prediction for the (i+1)th input
12:            loss ← LossFunction(y_hat, Y[i+1])
13:            batch_loss ← batch_loss + loss
14:        end for
15:    end for
16:    average_loss ← batch_loss / batch_size
17:    θ ← θ - η * gradient(average_loss with respect to θ)  // Perform a gradient update step
18: end for
19: return θ* ← θ

/* End Algorithm */

```
- **How the tasks on which in-context learning is successful**:

    It's not clear whether a model's ability to successfully perform a task in-context depends on having seen similar tasks or patterns during its training, or whether it relies on more general learning and reasoning abilities that are not directly tied to the specifics of the training data.

- **Theoretical Background**

## Results

### Linear Functions

The Transformer successfully approximates linear functions through in-context learning, emphasizing the precision of its approximations both globally (across different scales of query input) and locally (in terms of directional accuracy).

![image](./linear_regression_01.png)    

- **Robustness in Varied Scenarios**:

  The model maintains high performance under several challenging conditions.

  - **Skewed Covariance**: Demonstrates reasonable error rates when facing non-isotropic Gaussian distributions, indicating robustness to distributional mismatches.
  - **Noisy Linear Regression**: Exhibits resilience to added noise in output data, closely mirroring the performance of the optimal least squares estimator.
  - **Prompt Scaling**: Shows relative robustness to changes in the scale of inputs or weights, highlighting the model's capability to handle variations in data magnitude.
  - **Orthant and Orthogonal Mismatches**: Successfully approximates linear functions even when in-context and query inputs lie in different spaces, underscoring its ability to generalize across spatial discrepancies.

![image](./linear_regression_02.png)    


### Complex Functions

- **Sparse Linear Functions**: The study shows the Transformer nearly matches the performance of Lasso, a sparsity-leveraging estimator, highlighting the model's capability to recognize and utilize sparsity in data.

- **Decision Trees**: The Transformer outperforms traditional greedy tree learning and XGBoost algorithms, even when these algorithms are given additional information about input signs. This suggests that the Transformer can discover efficient algorithms for learning decision trees from the training distribution.

- **Two-Layer ReLU Neural Networks**: The Transformer shows comparable error rates to a baseline two-layer neural network trained on in-context examples using gradient descent. Moreover, the Transformer can in-context learn linear functions, demonstrating versatility and the ability to generalize across function classes.

![image](./complex_functions.png)

### Model Capacity and Problem Dimension

- **Importance of Model Capacity**: Increasing a model's capacity significantly enhances its ability to learn from in-context examples. This improvement is particularly notable when facing out-of-distribution prompts, demonstrating that a larger model is better equipped to generalize beyond the scenarios encountered during training.

- **Role of Problem Dimensionality**: The study highlights the interplay between problem dimensionality and model capacity. As the problem dimension (d) decreases or model capacity increases, in-context learning performance improves, suggesting a direct relationship between the complexity of the learning task and the resources required to tackle it effectively.

- **Performance Across Different Distributions**: The models show improved error rates with 2d in-context examples across both in-distribution and out-of-distribution prompts. This finding emphasizes the value of model capacity not just for familiar tasks but also for adapting to new challenges.

![image](./model_capacity_input_dimension.png)    

## Code Demo
A section providing sample code or a demonstration of the Transformer model tackling a simple function class. This could include a Jupyter notebook or a script that one can execute to see the model in action.

## Critical Analysis on Results
This section might delve into the performance metrics of the Transformer models' in-context learning, discussing the successes and limitations demonstrated by the experimental results, and comparing these to traditional learning algorithms.

## Implication
The paper discusses the broader implications of the findings, such as what they suggest about the nature of in-context learning and the potential for Transformers in various applications beyond the function classes tested.

## Resources
A compilation of additional reading materials, datasets, and tools relevant to in-context learning and Transformers. It might include links to pre-trained models, GitHub repositories, and supplementary material.

## Citation
Proper citation format for referencing the paper, which might follow a standard format such as APA, MLA, or IEEE to ensure that readers and other researchers can reference the work correctly. 


Can you make a few points and add explanations for my README file using the above content?



To determine if Transformers can predict outputs that meet the inequality condition, we typically follow these steps:

1. **Training the Model**: Before inference, the model is trained on a range of input-output pairs that reflect the diversity of the function class $F$. This training does not necessarily need to cover every possible function in $F$ but should be representative enough to enable the model to generalize.

2. **Creating the Prompt**: At inference time, a prompt is created that includes in-context examples. These are input-output pairs sampled from the function class $F$, along with a new query input. The model uses this prompt to infer the function that generated the in-context examples.

3. **Model Prediction**: Given the prompt, the model makes a prediction $ M(P) \) for the output corresponding to the query input \( x_{\text{query}} \).

4. **Calculating the Loss**: The predicted output is compared to the true output \( f(x_{\text{query}}) \) using a predefined loss function \( \ell \). The loss function quantifies the error between the model's prediction and the true output.

5. **Estimating the Expected Loss**: The average or expected loss is estimated by averaging the losses over a large number of prompts, each with a different set of in-context examples and query inputs, sampled from the distributions \( D_F \) and \( D_X \).

6. **Evaluating Against the Threshold \( \epsilon \)**: If the expected loss is less than or equal to the threshold \( \epsilon \), then the model is said to meet the condition for in-context learning of the function class \(F\).

7. **Statistical Significance**: To ensure that the results are not due to chance, statistical tests can be applied to determine the significance of the results. Tools from statistical learning theory, such as concentration inequalities (like Hoeffding's inequality), can be used to provide confidence that the observed performance is a true reflection of the model's capabilities.

This process would be described in detail in the methodology section of the paper or technical documentation and would be implemented in practice to test the model's in-context learning performance.



Limitation

Not talk about other function class.
How to combine all function classes in a model