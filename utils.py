import numpy as np


def compute_optimal_value_and_policy(T: int, X: int) -> tuple[np.ndarray, np.ndarray]:
    """Computes the optimal value function and policy matrices X x T

    Args:
        T (_type_): max time steps.
            The resulting time space will be {0, ..., T}
        X (_type_): max inventory space.
            The resulting inventory state space will be {0, ..., X}

    Returns: tuple[np.ndarray, np.ndarray]: optimal value and policy matrices
    """

    # define the value function matrix
    Vx = np.zeros((X, T))
    Vx[:, -1] = -0.1 * np.arange(X)  # set the boundary condition

    # define the optimal policy matrix
    policy = np.zeros((X, T), dtype=int)

    A = [0, 1]  # action space
    D_t = [0, 1]  # demand outcomes
    S_t = [0, 1]  # delivery outcomes

    # iterating backwards in time from T-1 to 0 inclusively
    for t in range(T - 2, -1, -1):
        # demand probability grows linearly in time
        # t+1 is since we are working with 0-based indexing
        P_demand = (t + 1) / T

        # calculate the Vt(x) for every state
        for x in range(X):
            Vxt = -np.inf
            best_action_xt = 0

            for a in A:
                # expected value for Vt(x) under action a
                Vxta = 0

                # probability of success depends on the action
                P_delivery_success = 0.5 if a == 1 else 0.0

                for d in D_t:
                    # probability on specific demand outcome
                    P_demand_d = P_demand if d == 1 else 1 - P_demand
                    for s in S_t:
                        # probability on specific delivery outcome, conditioned on the action
                        P_delivery_success_s = (
                            P_delivery_success if s == 1 else 1 - P_delivery_success
                        )

                        # immediate reward
                        r = min(x + a * s, d) - 0.1 * x

                        # simulated next state via transition function (capped at max inventory size)
                        x_next = min(max(0, x - d + a * s), X - 1)

                        # max expected value of the next state
                        v_next = Vx[x_next, t + 1]

                        Vxta += P_demand_d * P_delivery_success_s * (r + v_next)

                # update optimal value and policy for (x,t)
                if Vxta > Vxt:
                    Vxt = Vxta
                    best_action_xt = a

            Vx[x, t] = Vxt
            policy[x, t] = best_action_xt

    return Vx, policy


def simulate(
    optimal_policy: np.ndarray,
    starting_state: int,
    iter: int = 1000,
    random_state: int = 42,
) -> list:
    """Simulates the stochastic process

    Args:
        optimal_policy (np.ndarray): optimal policy matrix
        starting_state (int): starting state x
        iter (int, optional): How many simulation iterations to do. Defaults to 1000.
        random_state (int, optional): random state. Defaults to 42.

    Returns:
        list: a list of total rewards after each iter of the simulation
    """
    rng = np.random.default_rng(random_state)
    T = optimal_policy.shape[1]
    X = optimal_policy.shape[0]

    assert starting_state < X

    def _simulate():
        state = starting_state
        total_reward = 0
        for t in range(T):
            best_action = optimal_policy[state, t]

            d = rng.binomial(n=1, p=(t + 1) / T)
            s = rng.binomial(n=1, p=0.5) if best_action == 1 else 0

            total_reward += min(state + best_action * s, d) - 0.1 * state

            state = min(max(0, state - d + best_action * s), X - 1)

        # final holding costs for not selling the items off
        total_reward += -0.1 * state

        return total_reward

    rewards = [_simulate() for _ in range(iter)]

    return rewards
