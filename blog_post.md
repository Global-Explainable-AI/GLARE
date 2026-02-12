# Why Ranking Actions Locally Beats Learning Global Value Functions: Lessons from GABAR

*How a simple shift in perspective—from "how far am I from the goal?" to "which action looks best right now?"—leads to policies that generalize 8x beyond training size.*

---

## The Setup: Planning is Hard, and It Gets Harder Fast

Imagine you're organizing a warehouse. You have 10 packages, a few trucks, and a couple of airplanes. A classical planner can figure out the optimal delivery sequence in seconds. Now scale that to 30 packages across multiple cities. The same planner might run for hours—or never finish at all.

This is the fundamental scaling challenge in classical AI planning. The state space grows exponentially with the number of objects. Planning is NP-hard in most domains. Traditional planners use heuristic search, which works brilliantly on small problems but chokes on large ones.

The natural question: **can we learn planning strategies from small, solvable problems and apply them to large, unsolvable ones?**

This is exactly what our paper [*Graph Neural Network Based Action Ranking for Planning*](link-to-paper) addresses. We present GABAR—a system that trains on problems with 6-10 objects and successfully solves problems with 100+ objects, achieving 89% success rate on instances 8x larger than anything it saw during training.

---

## The Core Insight: Stop Trying to Learn the Hardest Thing

Most learning-based planning approaches try to learn a **value function** $V(s)$—an estimate of how far state $s$ is from the goal. The idea is simple: if you know $V(s)$ for every state, you can greedily pick the action that leads to the state with the lowest value.

The problem? This requires **global consistency**. Your value function must correctly rank *every reachable state* relative to *every other reachable state*. As the problem grows, the number of states explodes exponentially. Learning a globally consistent function over this space is itself an extremely hard problem—and in domains where optimal planning is NP-hard, there's no reason to believe such a function generalizes to larger instances.

Here's our key realization:

> **You don't need to rank all states. You only need to rank the actions available *right now*.**

At any given state in a typical planning problem, you might have 5-50 applicable actions. Ranking these is a *local* problem. You don't need to know anything about states three steps ahead. You just need to identify which of the currently available actions is most promising.

This is a fundamentally simpler learning target:
- **Value function learning**: Learn a function consistent across millions of states
- **Action ranking**: Learn to pick the best among a handful of local options

The difference matters enormously for generalization. Local patterns—"if a block is clear and needs to be somewhere else, pick it up"—tend to transfer across problem sizes. Global value relationships—"this specific state is exactly 17 steps from the goal"—do not.

---

## How GABAR Works: Three Ideas Working Together

GABAR combines three architectural choices that each address a specific challenge. Let me walk through each one.

### 1. Action-Centric Graph Representation

Most GNN-based planners represent a state as a graph of objects connected by predicates. If block A is on block B, there's an edge between them labeled "on."

We add something new: **action nodes**. Every applicable action in the current state gets its own node in the graph, connected to the objects it involves.

Why does this matter? Consider the action `unstack(A, B)`—pick up block A from block B. In our graph, this action has explicit edges to both A and B, with edge features encoding:
- Which parameter position each object fills (A is param 1, B is param 2)
- Which predicates each object satisfies in this action's context

This gives the GNN direct access to structured action information during message passing. The network doesn't have to *infer* action applicability from predicate patterns—it's explicitly represented.

**The ablation result is stark**: removing action nodes drops performance from 89% to 7% on hard problems. This single design choice accounts for the largest performance gain in our system.

### 2. GNN Encoder with Global Context

Our GNN processes the graph through 9 rounds of message passing. Each round updates edges, then nodes, then a global summary vector:

$$\mathbf{e}^{l+1}_{ij} = \phi_e([\mathbf{e}^l_{ij}; \mathbf{v}^l_i; \mathbf{v}^l_j; \mathbf{g}^l])$$

$$\mathbf{v}^{l+1}_i = \phi_v([\mathbf{v}^l_i; \text{AGG}(\{\mathbf{e}^{l+1}_{ij}\}); \mathbf{g}^l])$$

$$\mathbf{g}^{l+1} = \phi_g([\mathbf{g}^l; \text{AGG}(\{\mathbf{v}^{l+1}_i\}); \text{AGG}(\{\mathbf{e}^{l+1}_{ij}\})])$$

The **global node** $\mathbf{g}$ is crucial. As problems scale up, graphs get larger, but the number of GNN rounds stays fixed at 9. Without a global node, information from distant parts of the graph would never reach each other. The global node acts as a communication shortcut—every node reads from it and writes to it at every round.

Removing the global node cuts performance roughly in half on hard problems (89% → 42%).

### 3. Conditional GRU Decoder

Here's a subtlety that matters more than you'd expect. Consider the action `drive(truck1, cityA, cityB)`. Selecting `truck1` constrains which cities make sense. Selecting `cityA` as the origin further constrains `cityB`. Parameters are *interdependent*.

Our decoder uses a GRU (Gated Recurrent Unit) that builds actions sequentially:

1. Initialize hidden state from the global graph embedding
2. Score all action schemas → select one (e.g., "drive")
3. Update hidden state with selected action's embedding
4. Score all objects for parameter 1 → select one (e.g., "truck1")
5. Update hidden state with selected object's embedding
6. Score all objects for parameter 2 → select one (e.g., "cityA")
7. Continue until all parameters are filled

Each selection is conditioned on all previous selections through the GRU's hidden state. We use beam search (width 2) to maintain multiple candidates.

Removing conditional decoding drops performance from 89% to 60% on hard problems—and the effect is most pronounced in domains with complex inter-parameter dependencies like Logistics and Rovers.

---

## Results: What the Numbers Actually Mean

### Generalization That Actually Works

| Difficulty | GABAR | GPL (Value) | ASNets | GRAPL | OpenAI O3 | Gemini 2.5 |
|:-----------|:-----:|:-----------:|:------:|:-----:|:---------:|:----------:|
| Easy       | 95.5% | 79.1%       | 76.0%  | 43.5% | 33.4%     | 44.0%      |
| Medium     | 92.2% | 28.5%       | 65.4%  | 29.3% | 11.6%     | 17.1%      |
| Hard       | 89.2% | 6.5%        | 48.5%  | 22.1% | 0.4%      | 1.5%       |

The coverage drop from easy to hard for GABAR is minimal: 95.5% → 89.2%. Compare this to GPL (79% → 6.5%) or state-of-the-art LLMs (33-44% → 0.4-1.5%).

On Blocks World, Gripper, and Miconic, GABAR achieves **100% success rate at all difficulty levels**—solving 40-block, 100-ball, and 100-passenger problems after training on instances with fewer than 10 objects.

### Plan Quality, Not Just Coverage

GABAR doesn't just solve more problems—it solves them *well*. The Plan Quality Ratio (plan length from Fast Downward / plan length from GABAR) stays at ~1.0 across all difficulties, meaning GABAR's plans are comparable in length to those from a state-of-the-art satisficing planner. On several domains, GABAR actually produces *shorter* plans than Fast Downward's LAMA configuration.

### The LLM Comparison

We tested OpenAI's O3 and Gemini 2.5 Pro using one-shot prompting. Both essentially collapse on hard problems (0.4% and 1.5% coverage). This isn't surprising—LLMs lack the structural inductive bias needed for systematic relational reasoning over large state spaces. They can pattern-match small planning problems from training data but cannot compose solutions for novel large instances.

---

## The Deeper Lessons: Invariants for Other Research

Beyond the specific results, GABAR demonstrates several principles that apply broadly.

### 1. Local Objectives Can Beat Global Ones

The most powerful lesson: you often don't need to learn the globally optimal function. If your downstream task only requires *local decisions*, formulate your learning objective locally.

This applies far beyond planning:
- **Recommendation systems**: Rank the items on *this page*, don't learn absolute item values
- **Dialogue systems**: Rank the next *response candidates*, don't model the entire conversation value
- **Compiler optimization**: Rank the transformations applicable *now*, don't estimate total program quality

The mathematical intuition: a local ranking function needs to be consistent only within each decision point's option set. A global value function needs consistency across *all* possible inputs. The former is a strictly easier learning problem.

### 2. Represent What You're Deciding About

GABAR's largest performance gain comes from explicitly representing actions in the input. This seems obvious in retrospect: if you want to rank actions, give the network direct access to action structure.

More generally: **your input representation should explicitly encode the entities you're making decisions about**. If you're selecting among candidate programs, represent program structure. If you're choosing among robot trajectories, represent trajectory features. Don't make the network reconstruct this information from indirect signals.

### 3. Structure Your Decoder to Match Your Output Structure

Actions have structure—a schema and ordered parameters with dependencies. Our GRU decoder respects this structure by building actions sequentially, conditioning each choice on previous ones.

The principle: **if your output has compositional structure, decode it compositionally**. This is why autoregressive language models work for text, why graph-to-sequence models work for molecules, and why our conditional decoder works for planning actions.

### 4. Global Context Nodes Enable Fixed-Depth Architectures to Scale

The global node is a simple idea with outsized impact. It lets a fixed-depth GNN (9 layers) process arbitrarily large graphs by providing a "shortcut" for information flow.

This pattern appears in many architectures:
- [CLS] tokens in transformers
- Global pooling in graph networks
- Memory cells in neural Turing machines

If your architecture has fixed depth but variable-size inputs, consider adding an explicit global aggregation mechanism.

### 5. Train on Easy, Deploy on Hard

GABAR is trained exclusively on problems that are trivial for classical planners (solved in milliseconds). The training data is essentially free—no human labeling, no expensive computation, just run a planner on small instances.

This "easy instances as training signal" paradigm works when:
- The underlying patterns are **compositional** (small-scale structure composes into large-scale behavior)
- Your architecture has appropriate **inductive biases** (GNNs handle variable-size relational inputs)
- Your learning objective **doesn't fight scaling** (local ranking vs. global values)

---

## What This Means Going Forward

### For the Planning Community

GABAR shows that learned policies can be practical for large planning problems. The 89% success rate on hard instances—combined with plan quality matching classical planners—suggests that learned policies are ready to be taken seriously as planning tools, not just research curiosities.

The approach also complements classical planners rather than replacing them: GABAR uses planners to generate training data, then handles the problems those planners can't solve in reasonable time.

### For the ML Community

The action ranking vs. value learning comparison is a concrete case study in how reformulating the learning objective—without changing the training data or model capacity—can dramatically improve generalization. This is a reminder that the choice of *what* to learn matters as much as *how* to learn it.

### For Anyone Building Systems That Generalize

The combination of structural representation + local objectives + compositional decoding is a recipe that extends beyond planning. Any domain where:
- Inputs are relational and variable-sized
- Decisions are local (choosing among current options)
- Outputs have compositional structure

...is a candidate for this approach. Molecular design, program synthesis, robotic task planning, network optimization—the pattern applies widely.

---

## Technical Details (For Those Who Want Them)

- **Training**: Adam optimizer, lr = 0.0005, batch size 16, hidden dim 64
- **Architecture**: 9 GNN rounds, beam width 2, attention-based aggregation
- **Data**: ~3,000-7,000 training examples per domain, generated by solving random small PDDL instances
- **Training time**: 1-2 hours per domain on a single RTX 3080
- **Evaluation**: 8 standard planning benchmarks (Blocks, Gripper, Miconic, Spanner, Logistics, Rovers, Visitall, Grid)
- **Cycle avoidance**: Maintains visited state history; falls back to next-ranked action if top choice leads to visited state

---

## Summary

GABAR demonstrates that a simple conceptual shift—from global value functions to local action ranking—combined with the right structural inductive biases, enables learned planning policies that genuinely generalize. The system trains on toy problems and solves real ones, maintains high plan quality as it scales, and substantially outperforms both classical learning baselines and state-of-the-art LLMs.

The broader takeaway: when you're building a system that needs to generalize, ask yourself—am I trying to learn something harder than I need to? Can I reformulate my objective to be *local* rather than *global*? Can I represent my decision space *explicitly* rather than *implicitly*? Can I decode my outputs *compositionally* rather than *monolithically*?

If the answer to any of these is yes, you might be working harder than necessary.

---

*This work was presented at NeurIPS 2025. Paper, code, and project page available at [project website link].*

*Supported by the Army Research Office under grant W911NF2210251.*
