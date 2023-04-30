Download Link: https://assignmentchef.com/product/solved-csce790-002-deep-reinforcement-learning-and-search-homework-2
<br>
<h1>Installation</h1>

We will be using the same conda environment as in Homework 1.

Keep in mind that this assignment relies on a correct implementation of assignments_code/assignment1.py, which is provided on Blackboard.

The entire GitHub repository should be downloaded again as changes were made to other files. You can download it with the green “Code” button and click “Download ZIP”. After downloading, you can replace assignment1.py with your own implementation from Homework 1 or with the implementation of assignment1.py found on Blackboard under Course Content – Homework – Homework 1 – Homework1 code answers.

<h1>1             Policy Iteration</h1>

<h2>1.1           Policy Iteration Implementation (30 pts)</h2>

Implement policy_evaluation_step (shown in Algorithm 2) in assignments_code/assignment2.py.

<strong>Key building blocks:</strong>

<ul>

 <li>state_action_dynamics(state, action): returns, in this order, the expected return <em>r</em>(<em>s,a</em>), all possible next states given the current state and action, their probabilities. Keep in mind, this only returns states that have a non-zero state-transition probability.</li>

 <li>get_actions() function that returns a list of all possible actions</li>

 <li>policy: you can obtain <em>π</em>(<em>a</em>|<em>s</em>) with policy[state][action]</li>

</ul>

Switches:

<ul>

 <li>–discount, to change the discount (default=1.0)</li>

 <li>–rand_right, to change the probability that the wind blows you to the right (default=0.0)</li>

 <li>–wait, the number of seconds to wait after every iteration of policy <em>iteration </em>so that you can visualize your algorithm (default=0.0)</li>

 <li>–wait_eval, the number of seconds to wait after every iteration of policy <em>evaluation </em>so that you can visualize your algorithm (default=0.0)</li>

</ul>

<strong>Running the code: </strong>python run_policy_iteration.py –map maps/map1.txt –wait 1.0

Policy iteration can be used to find, or approximate, the optimal value function. Policy iteration starts with a given value and policy function and iterates between policy evaluation and policy improvement until the policy function stops changing. For more information, see the lecture slides or Chapter 4 of Sutton and Barto. In this exercise, we will be finding the optimal value function, exactly, using policy iteration.

We will start with a uniform random policy and a value function that is zero at all states. The policy improvement step has already been implemented using the get_action function that you implemented in Homework 1. Feel free to use your current implementation or the solution provided to you. Therefore, you only have to implement policy_evaluation_step.

Vary the environment dynamics by making the environment stochastic –rand_right to 0.1 and 0.5. Compare your results to the solution videos provided in Blackboard under Course Content – Homework Homework 2.

<strong>Algorithm 1 </strong>Policy Evaluation

1: <strong>procedure </strong>Policy Evaluation(S<em>,V,π,γ,θ</em>)

2:              ∆ ← inf

3:               <strong>while </strong>∆ <em>&gt; θ </em><strong>do</strong>

4:                          ∆<em>,V </em>= Policy Evaluation Step(S<em>,V,π,γ</em>)

5:             <strong>end while</strong>

<table width="720">

 <tbody>

  <tr>

   <td width="566">6:              <strong>return </strong><em>V</em>7: <strong>end procedure</strong></td>

   <td width="154"><em>. </em>Approximation of <em>v<sub>π</sub></em></td>

  </tr>

  <tr>

   <td width="566"> </td>

   <td width="154"> </td>

  </tr>

  <tr>

   <td width="566"><strong>Algorithm 2 </strong>Policy Evaluation Step</td>

   <td width="154"> </td>

  </tr>

  <tr>

   <td width="566">1: <strong>procedure </strong>Policy Evaluation Step(S<em>,V,π,γ</em>)2:              ∆ ← 03:               <strong>for </strong><em>s </em>∈ S <strong>do</strong>4:                      <em>v </em>← <em>V </em>(<em>s</em>)5:                           <em>V </em>(<em>s</em>) ← <sup>P</sup><em><sub>a </sub>π</em>(<em>a</em>|<em>s</em>)(<em>r</em>(<em>s,a</em>) + <em>γ </em><sup>P</sup><em><sub>s</sub></em>0 <em>p</em>(<em>s</em><sup>0</sup>|<em>s,a</em>)<em>V </em>(<em>s</em><sup>0</sup>))6:                        ∆ ← max(∆<em>,</em>|<em>v </em>− <em>V </em>(<em>s</em>)|)7:             <strong>end for</strong>8:               <strong>return </strong>∆<em>,V</em>9: <strong>end procedure</strong></td>

   <td width="154"> </td>

  </tr>

 </tbody>

</table>

<h2>1.2           Policy Iteration Concept (20 pts)</h2>

For policy evaluation, we started with a uniform random policy, a discount of <em>γ </em>= 1, and stopped policy evaluation only when ∆ = 0. If we started, instead, with a policy that gave the same action in every state (i.e. in every state, go left) we would never converge to ∆ = 0. Why is this? If we reduce <em>γ</em>, we would converge to ∆ = 0. Why is this?

<h1>2             Q-learning</h1>

<h2>2.1           Q-learning Implementation (30 pts)</h2>

Implement q_learning_step (shown in Algorithm 4) in assignments_code/assignment2.py.

<strong>Key building blocks:</strong>

<ul>

 <li>sample_transition(state, action): returns, in this order, the next state and reward</li>

 <li>get_actions() function that returns a list of all possible actions</li>

 <li>action_vals: you can obtain <em>Q</em>(<em>s,a</em>) with action_vals[state][action]</li>

</ul>

Switches:

<ul>

 <li>–discount, to change the discount (default=1.0)</li>

 <li>–rand_right, to change the probability that the wind blows you to the right (default=0.0)</li>

 <li>–wait_greedy, Your learned greedy policy is visualized every 100 episodes for 40 steps. This is the number of seconds to wait after each step (default=0.1)</li>

 <li>–wait_step, the number of seconds to wait after every step of Q-learning so that you can visualize your algorithm (default=0.0)</li>

</ul>

<strong>Running the code: </strong>python run_q_learning.py –map maps/map1.txt –wait_greedy 0.1

Q-learning is a model-free, off-policy, temporal difference algorithm. Q-learning follows an -greedy behavior policy and evaluates a greedy target policy. An -greedy policy takes a random action with probability , otherwise it takes the greedy action, argmax<em><sub>a </sub>Q</em>(<em>S,a</em>).

In this setting, we will be running Q-learning with a learning rate <em>α </em>= 0<em>.</em>5 and 1. Each episode ends when the agent reaches the goal. We will run Q-learning for 1000 episodes. For more information, see the lecture slides or Chapters 5 and 6 of Sutton and Barto.

<strong>Algorithm 3 </strong>Q-learning

1: <strong>procedure </strong>Q-learning

2:               <strong>for </strong><em>i </em>∈ 1<em>…N </em><strong>do</strong>

3:                     Initialize <em>S</em>

4:                      <strong>while </strong><em>S </em>is not terminal <strong>do</strong>

5:                                <em>S,Q </em>= Q Learning Step(

6:                     <strong>end while</strong>

7:             <strong>end for</strong>

8:              <strong>return </strong><em>Q                                                                                                                . </em>Approximation of <em>q</em><sub>∗</sub>

9: <strong>end procedure</strong>

2:                Sample an action <em>A </em>from -greedy policy derived from <em>Q</em>

3:                      <em>S</em><sup>0</sup><em>,R </em>= env.sample_transition(S,A)

4:                       <em>Q</em>(<em>S,A</em>) = <em>Q</em>(<em>S,A</em>) + <em>α</em>(<em>R </em>+ <em>γ </em>max<em><sub>a </sub>Q</em>(<em>S</em><sup>0</sup><em>,a</em>) − <em>Q</em>(<em>S,A</em>))

5:               <strong>return </strong><em>S</em><sup>0</sup><em>,Q</em>

6: <strong>end procedure</strong>

Vary the environment dynamics by making the environment stochastic with –rand_right to 0.1 and 0.5. Compare your results to the solution videos provided in Blackboard under Course Content – Homework Homework 2. Keep in mind that this algorithm is stochastic, so results will not be exactly the same.

<h2>2.2           Q-learning Concept (20 pts)</h2>

In the case where we do Q-learning with –rand_right set to 0.5, the actions in the states in the top left remain bright green, indicating that these states are, relatively, the best states. However, we know from finding the optimal value function via policy iteration that this is not the case. Why does this occur and why do we not have this problem when –rand_right set to 0.5 when doing value iteration or policy iteration?

<h1>What to Turn In</h1>

Turn in your implementation of assignments_code/assignment2.py and a PDF of your answer to question parts 1.2 and 2.2.