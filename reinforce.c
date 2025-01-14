#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "grad/grad.h"
#include "sim/sim.h"

#define NUM_EPISODES 1000
#define MAX_STEPS 1000
#define GAMMA 0.99
#define LEARNING_RATE 0.001

#define DT_PHYSICS (1/1000)

// Define the state and action dimensions
#define STATE_DIM 12  // Position (3), Velocity (3), Orientation (3), Angular Velocity (3)
#define ACTION_DIM 4  // Rotor speeds (4)

// Reward function: Encourages the quadcopter to stay close to the desired hover position
double compute_reward(double* state, double* target_position) {
    double position_error = 0.0;
    for (int i = 0; i < 3; i++) {
        position_error += pow(state[i] - target_position[i], 2);
    }
    return -position_error;  // Negative error to encourage minimization
}

// Policy network: Takes state as input and outputs action probabilities
double* policy_forward(Net* net, double* state) {
    double** act = malloc(5 * sizeof(double*));
    for (int i = 0; i < 5; i++) {
        act[i] = malloc(net->sz[i] * sizeof(double));
    }
    fwd(net, state, act);
    double* action_probs = malloc(ACTION_DIM * sizeof(double));
    memcpy(action_probs, act[4], ACTION_DIM * sizeof(double));
    for (int i = 0; i < 5; i++) free(act[i]);
    free(act);
    return action_probs;
}

// Sample action from the policy network's output probabilities
int sample_action(double* action_probs) {
    double r = (double)rand() / RAND_MAX;
    double sum = 0.0;
    for (int i = 0; i < ACTION_DIM; i++) {
        sum += action_probs[i];
        if (r < sum) return i;
    }
    return ACTION_DIM - 1;
}

// REINFORCE algorithm: Update the policy network using the collected trajectories
void reinforce_update(Net* net, double** states, double** actions, double* rewards, int num_steps) {
    double* discounted_rewards = malloc(num_steps * sizeof(double));
    double cumulative_reward = 0.0;
    for (int i = num_steps - 1; i >= 0; i--) {
        cumulative_reward = rewards[i] + GAMMA * cumulative_reward;
        discounted_rewards[i] = cumulative_reward;
    }

    double** act = malloc(5 * sizeof(double*));
    double** grad = malloc(5 * sizeof(double*));
    for (int i = 0; i < 5; i++) {
        act[i] = malloc(net->sz[i] * sizeof(double));
        grad[i] = malloc(net->sz[i] * sizeof(double));
    }

    for (int i = 0; i < num_steps; i++) {
        double* action_probs = policy_forward(net, states[i]);
        double action_prob = action_probs[(int)actions[i][0]];
        double log_prob = log(action_prob);
        double advantage = discounted_rewards[i];

        // Compute gradient
        for (int j = 0; j < ACTION_DIM; j++) {
            grad[4][j] = (j == (int)actions[i][0]) ? -advantage / action_prob : 0.0;
        }
        bwd(net, act, grad[4], grad, 0.0);

        free(action_probs);
    }

    for (int i = 0; i < 5; i++) { free(act[i]); free(grad[i]); }
    free(act); free(grad);
    free(discounted_rewards);
}

int main() {
    srand(time(NULL));

    // Initialize the quadcopter simulation
    Sim* sim = init_sim(false);
    double target_position[3] = {0.0, 1.0, 0.0};  // Desired hover position

    // Initialize the policy network
    int sz[] = {STATE_DIM, 128, 64, 32, ACTION_DIM};
    Net* policy_net = init_net(5, sz);

    // Training loop
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        reset_quad(sim->quad, 0.0, 1.0, 0.0);  // Reset quadcopter to initial position

        double** states = malloc(MAX_STEPS * sizeof(double*));
        double** actions = malloc(MAX_STEPS * sizeof(double*));
        double* rewards = malloc(MAX_STEPS * sizeof(double));

        int num_steps = 0;
        double total_reward = 0.0;

        for (int step = 0; step < MAX_STEPS; step++) {
            // Get current state
            double* state = malloc(STATE_DIM * sizeof(double));
            memcpy(state, sim->quad->linear_position_W, 3 * sizeof(double));
            memcpy(state + 3, sim->quad->linear_velocity_W, 3 * sizeof(double));
            memcpy(state + 6, sim->quad->R_W_B, 3 * sizeof(double));
            memcpy(state + 9, sim->quad->angular_velocity_B, 3 * sizeof(double));

            // Get action from policy network
            double* action_probs = policy_forward(policy_net, state);
            int action = sample_action(action_probs);
            double* action_vec = malloc(ACTION_DIM * sizeof(double));
            action_vec[0] = (double)action;

            // Execute action in the simulation
            sim->quad->omega_next[action] = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * action_probs[action];
            update_quad(sim->quad, DT_PHYSICS);

            // Compute reward
            double reward = compute_reward(state, target_position);
            total_reward += reward;

            // Store state, action, and reward
            states[num_steps] = state;
            actions[num_steps] = action_vec;
            rewards[num_steps] = reward;
            num_steps++;

            // Print progress every 100 steps
            if (step % 100 == 0) {
                printf("Step %d: Position (%.2f, %.2f, %.2f), Reward: %.2f\n",
                       step, sim->quad->linear_position_W[0],
                       sim->quad->linear_position_W[1],
                       sim->quad->linear_position_W[2],
                       reward);
            }

            free(action_probs);
        }

        // Update policy network using REINFORCE
        reinforce_update(policy_net, states, actions, rewards, num_steps);

        // Print episode summary
        double avg_reward = total_reward / num_steps;
        printf("Episode %d: Average Reward = %.2f\n", episode, avg_reward);

        // Free memory
        for (int i = 0; i < num_steps; i++) {
            free(states[i]);
            free(actions[i]);
        }
        free(states);
        free(actions);
        free(rewards);

        // Periodically test the policy
        if (episode % 10 == 0) {
            printf("Testing Policy...\n");
            reset_quad(sim->quad, 0.0, 1.0, 0.0);
            double test_reward = 0.0;
            for (int step = 0; step < MAX_STEPS; step++) {
                double* state = malloc(STATE_DIM * sizeof(double));
                memcpy(state, sim->quad->linear_position_W, 3 * sizeof(double));
                memcpy(state + 3, sim->quad->linear_velocity_W, 3 * sizeof(double));
                memcpy(state + 6, sim->quad->R_W_B, 3 * sizeof(double));
                memcpy(state + 9, sim->quad->angular_velocity_B, 3 * sizeof(double));

                double* action_probs = policy_forward(policy_net, state);
                int action = 0;
                double max_prob = action_probs[0];
                for (int i = 1; i < ACTION_DIM; i++) {
                    if (action_probs[i] > max_prob) {
                        action = i;
                        max_prob = action_probs[i];
                    }
                }
                sim->quad->omega_next[action] = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * action_probs[action];
                update_quad(sim->quad, DT_PHYSICS);

                double reward = compute_reward(state, target_position);
                test_reward += reward;

                free(state);
                free(action_probs);
            }
            printf("Test Reward: %.2f\n", test_reward / MAX_STEPS);
        }

        // Save the policy every 100 episodes
        if (episode % 100 == 0) {
            char filename[100];
            sprintf(filename, "policy_episode_%d.bin", episode);
            save_weights(policy_net, filename);
        }
    }

    // Clean up
    free_net(policy_net);
    free_sim(sim);

    return 0;
}