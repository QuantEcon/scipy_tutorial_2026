---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Schelling Model with NumPy

*Prepared for SciPy 2026*

**Authors: [John Stachurski](https://johnstachurski.net/), [Thomas J. Sargent](http://www.tomsargent.com/), [Smit Lunagariya](https://smit-create.github.io/intro.html), [Matt McKay](https://github.com/mmcky)**

## Overview

In the [previous lecture](https://quantecon.github.io/scipy_tutorial_2026/schelling.html), we implemented the Schelling
segregation model using pure Python and standard libraries, rather than
Python plus numerical and scientific libraries.

In this lecture, we rewrite the model using NumPy arrays and functions.

NumPy is the most fundamental library for numerical coding in Python.

We’ll achieve greater speed!!

```{code-cell} ipython3
:hide-output: false

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import time
```

## Data Representation

In the class-based version, each agent was an object storing its own type and location.

Here we take a different approach: we store all agent data in NumPy arrays.

- `locations` — an $ n \times 2 $ array where row $ i $ holds the $ (x, y) $ coordinates of agent $ i $
- `types` — an array of length $ n $ where entry $ i $ is 0 or 1, indicating agent $ i $’s type


Let’s set up the parameters:

```{code-cell} ipython3
:hide-output: false

num_of_type_0 = 1000    # number of agents of type 0 (orange)
num_of_type_1 = 1000    # number of agents of type 1 (green)
n = num_of_type_0 + num_of_type_1  # total number of agents
num_neighbors = 10      # number of agents viewed as neighbors
max_other_type = 6      # max number of different-type neighbors tolerated
```

Here’s a function to initialize the state with random locations and types:

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

def initialize_state(rng):
    pass
```

Let’s see what this looks like:

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

rng = default_rng(1234)
locations, types = initialize_state(rng)

print(f"locations shape: {locations.shape}")
print(f"First 5 locations:\n{locations[:5]}")
print(f"\ntypes shape: {types.shape}")
```

## Helper Functions

Let’s write some functions that compute what we need while operating on the arrays.

+++

### Computing Distances

To find an agent’s neighbors, we need to compute distances.

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

def get_distances(loc, locations):
    """
    Compute the Euclidean distance from one location to all agent locations.
    """
    pass
```

Let’s break down how this function works:

1. `loc - locations` subtracts the reference point `loc` from every row of
  `locations`. NumPy “broadcasts” the subtraction, so if `loc = [0.5, 0.3]`
  and `locations` has 2000 rows, the result is a 2000 × 2 array where each
  row is the difference vector from `loc` to that agent.
1. `np.linalg.norm(..., axis=1)` computes the Euclidean norm of each
  row. The `axis=1` argument tells NumPy to compute the norm across columns
  (i.e., for each row separately).


This vectorized approach is much faster than looping through agents one by one.

+++

### Finding Neighbors

Now we can find the nearest neighbors:

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

def get_neighbors(i, locations):
    """
    Get indices of the nearest neighbors to agent i (excluding self).
    """
    pass
```

Here’s how this function works:

1. First we call `get_distances` to get an array of 2000 distances (one for
  each agent).
1. We set `distances[i] = np.inf` so that agent $ i $ doesn’t count as their own
  neighbor.
1. `np.argsort(distances)` returns the *indices* of agents sorted from closest
  to furthest. For example, if the closest agent has index
  42, then `indices[0]` equals 42.
1. `indices[:num_neighbors]` uses slicing to keep only the first `num_neighbors`
  indices — these correspond to the nearest neighbors.

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

# Find neighbors of agent 0
neighbors = get_neighbors(0, locations)
print(f"Agent 0's nearest neighbors: {neighbors}")
print(f"Agent 0 is NOT included: {0 not in neighbors}")
```

### Checking Happiness

An agent is happy if enough of their neighbors share their type:

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

def is_happy(i, locations, types):
    """
    True if agent i has no more than max_other_type neighbors of a different type.
    """
    pass
```

Let’s walk through this function step by step:

1. `types[i]` gets the type (0 or 1) of agent $ i $.
1. `get_neighbors(i, locations)` returns an array of indices for the nearest
  neighbors.
1. `types[neighbors]` uses these indices to look up the types of the
  neighbors. This is called “fancy indexing” — when you pass an array of
  indices to another array, NumPy returns the elements at those positions.
  For example, if `neighbors = [42, 7, 15, ...]`, then `types[neighbors]`
  returns `[types[42], types[7], types[15], ...]`.
1. `neighbor_types == agent_type` compares each neighbor’s type to the agent’s
  type, producing an array of `True`/`False` values (e.g.,
  `[True, False, True, ...]`).
1. `np.sum(...)` counts how many `True` values there are. In NumPy, `True`
  is treated as 1 and `False` as 0, so summing a boolean array counts the
  `True` entries.
1. Finally, we check if this count is within the tolerance `max_other_type`.

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

# Check if agent 0 is happy
print(f"Agent 0 type: {types[0]}")
print(f"Agent 0 happy: {is_happy(0, locations, types)}")
```

### Counting Happy Agents

The next function uses a loop to check each agent and count how many are happy.

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

def count_happy(locations, types):
    """
    Count the number of happy agents.
    """
    pass
```

Since `is_happy` returns `True` or `False`, and Python treats `True`
as 1 when adding, we can accumulate the count directly.

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

print(f"Initially happy agents: {count_happy(locations, types)} out of {n}")
```

### Moving Unhappy Agents

When an agent is unhappy, they keep trying new random locations until they find
one where they’re happy:

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

def update_agent(i, locations, types, rng, max_attempts=10_000):
    """
    Move agent i to a new location where they are happy.
    """
    pass
```

Here’s how this works:

1. The `while` loop keeps running as long as the agent is unhappy.
1. `locations[i, :] = uniform(), uniform()` assigns a new random $ (x, y) $
  location to agent $ i $. The left side `locations[i, :]` selects row $ i $
  (all columns), and the right side creates a tuple of two random numbers
  between 0 and 1.
1. Importantly, this modifies the `locations` array *in place*. We don’t need
  to return anything because the original array is changed directly. This is
  a key feature of NumPy arrays — when you modify a slice, you modify the
  underlying data.

+++

## Visualization

Here’s some code for visualization — we’ll skip the details

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

def plot_distribution(locations, types, title):
    " Plot the distribution of agents. "
    fig, ax = plt.subplots()
    plot_args = {
        'markersize': 6, 'alpha': 0.8,
        'markeredgecolor': 'black',
        'markeredgewidth': 0.5
    }
    colors = 'darkorange', 'green'
    for agent_type, color in zip((0, 1), colors):
        idx = (types == agent_type)
        ax.plot(locations[idx, 0],
                locations[idx, 1],
                'o',
                markerfacecolor=color,
                **plot_args)
    ax.set_title(title)
    plt.show()
```

Let’s visualize the initial random distribution:

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

rng = default_rng(1234)
locations, types = initialize_state(rng)
plot_distribution(locations, types, 'Initial random distribution')
```

## The Simulation

Now we put it all together.

As in the first lecture, each iteration cycles through all agents in order,
giving each the opportunity to move:

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

def run_simulation(max_iter=100_000, seed=42):
    """
    Run the Schelling simulation.
    Each iteration cycles through all agents, giving each a chance to move.
    """
    rng = default_rng(seed)
    locations, types = initialize_state(rng)

    plot_distribution(locations, types, 'Initial distribution')

    pass
```

## Results

Let’s run the simulation:

```{code-cell} ipython3
:hide-output: false
:tags: [skip-execution]

locations, types = run_simulation()
```

We see the same phenomenon as in the class-based version: starting from a
random mixed distribution, agents self-organize into segregated clusters.

+++

## Exercise

Implement a quantitative measure of segregation. The **exposure index** measures, on average, what fraction of an agent's nearest neighbours are of the *same* type.

Write `segregation_index(locations, types)` that returns a float between 0 and 1, where 1 means all agents' nearest neighbours are the same type (full segregation) and 0.5 means fully mixed. Run it on the initial random distribution and again after the simulation and compare the values.

```{code-cell} ipython3
:tags: [skip-execution]

def segregation_index(locations, types):
    """
    Compute the mean same-type neighbor fraction across all agents.

    Returns a value in [0, 1]:
      ~0.5  →  fully mixed (each agent has ~50% same-type neighbors)
      ~1.0  →  fully segregated (all neighbors same type)
    """
    pass


rng = default_rng(1234)
locations, types = initialize_state(rng)

idx_before = segregation_index(locations, types)
print(f"Segregation index before simulation: {idx_before:.3f}")

locations, types = run_simulation()

idx_after = segregation_index(locations, types)
print(f"Segregation index after simulation:  {idx_after:.3f}")
print(f"Change: +{idx_after - idx_before:.3f}")
```
