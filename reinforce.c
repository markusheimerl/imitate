#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "mlp/mlp.h"
#include "sim/quad.h"

// Constants for the RL environment
#define STATE_DIM 9        // position(3) + velocity(3) + euler_angles(3)
#define ACTION_DIM 4       // rotor commands
#define HIDDEN_DIM 512     // policy network hidden layer size
#define MAX_STEPS 1000     // maximum steps per episode
#define DT 0.02           // control timestep (20ms)

#define DT_PHYSICS  (1.0 / 1000.0)

// Environment structure
typedef struct {
    Quad* quad;           // Quadcopter state
    double target[3];     // Target position
    double state[STATE_DIM]; // Current state vector
    double max_dist;      // Maximum allowed distance from target
    int steps;           // Current episode step counter
} Environment;

// Helper function for Gaussian noise
double randn() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Initialize environment
Environment* create_environment() {
    Environment* env = (Environment*)malloc(sizeof(Environment));
    env->quad = create_quad(0.0, 0.0, 0.0);
    env->max_dist = 5.0;
    env->steps = 0;
    return env;
}

// Reset environment to initial state
void environment_reset(Environment* env) {
    // Random initial position near origin
    double init_x = (double)rand() / RAND_MAX * 0.2 - 0.1;
    double init_y = (double)rand() / RAND_MAX * 0.2;
    double init_z = (double)rand() / RAND_MAX * 0.2 - 0.1;
    
    // Random target position
    env->target[0] = (double)rand() / RAND_MAX * 4.0 - 2.0;
    env->target[1] = 1.5;
    env->target[2] = (double)rand() / RAND_MAX * 4.0 - 2.0;
    
    // Reset quad state
    reset_quad(env->quad, init_x, init_y, init_z);
    env->steps = 0;
}

// Convert quad state to observation vector
void state_from_quad(Environment* env) {
    // Position
    env->state[0] = env->quad->linear_position_W[0];
    env->state[1] = env->quad->linear_position_W[1];
    env->state[2] = env->quad->linear_position_W[2];
    
    // Velocity
    env->state[3] = env->quad->linear_velocity_W[0];
    env->state[4] = env->quad->linear_velocity_W[1];
    env->state[5] = env->quad->linear_velocity_W[2];
    
    // Convert rotation matrix to euler angles
    env->state[6] = atan2(env->quad->R_W_B[7], env->quad->R_W_B[8]); // roll
    env->state[7] = asin(-env->quad->R_W_B[6]);                      // pitch
    env->state[8] = atan2(env->quad->R_W_B[3], env->quad->R_W_B[0]); // yaw
}

// Compute reward based on current state
double compute_reward(Environment* env) {
    double pos_error = 0.0;
    for(int i = 0; i < 3; i++) {
        double diff = env->quad->linear_position_W[i] - env->target[i];
        pos_error += diff * diff;
    }
    
    // Compute velocity penalty
    double vel_penalty = 0.0;
    for(int i = 0; i < 3; i++) {
        vel_penalty += env->quad->linear_velocity_W[i] * 
                      env->quad->linear_velocity_W[i];
    }
    
    // Negative reward (smaller is better)
    return -(sqrt(pos_error) + 0.1 * sqrt(vel_penalty));
}

// Check if episode should terminate
bool is_terminal(Environment* env) {
    // Check if max steps reached
    if(env->steps >= MAX_STEPS) return true;
    
    // Check if quad is too far from target
    double dist = 0.0;
    for(int i = 0; i < 3; i++) {
        double diff = env->quad->linear_position_W[i] - env->target[i];
        dist += diff * diff;
    }
    if(sqrt(dist) > env->max_dist) return true;
    
    // Check if quad hit the ground
    if(env->quad->linear_position_W[1] <= 0.0) return true;
    
    return false;
}

// Take a step in the environment
double environment_step(Environment* env, double* action) {
    // Apply action to quad
    for(int i = 0; i < 4; i++) {
        env->quad->omega[i] = fmax(OMEGA_MIN, 
                                 fmin(OMEGA_MAX, action[i]));
    }
    
    // Simulate physics for one control timestep
    // (might need multiple physics steps per control step)
    int physics_steps = (int)(DT / DT_PHYSICS);
    for(int i = 0; i < physics_steps; i++) {
        update_quad(env->quad, DT_PHYSICS);
    }
    
    // Update state
    state_from_quad(env);
    env->steps++;
    
    // Compute reward
    return compute_reward(env);
}

// Free environment
void free_environment(Environment* env) {
    free(env->quad);
    free(env);
}

// Episode data storage
typedef struct {
    double* states;      // [MAX_STEPS x STATE_DIM]
    double* actions;     // [MAX_STEPS x ACTION_DIM]
    double* means;       // [MAX_STEPS x ACTION_DIM]
    double* rewards;     // [MAX_STEPS]
    double* returns;     // [MAX_STEPS]
    int length;         // Actual episode length
} Episode;

// REINFORCE hyperparameters
typedef struct {
    double learning_rate;
    double gamma;        // Discount factor
    double sigma;        // Fixed policy standard deviation
    int hidden_dim;
    int batch_size;
} RLParams;

// Create episode storage
Episode* create_episode() {
    Episode* ep = (Episode*)malloc(sizeof(Episode));
    ep->states = (double*)malloc(MAX_STEPS * STATE_DIM * sizeof(double));
    ep->actions = (double*)malloc(MAX_STEPS * ACTION_DIM * sizeof(double));
    ep->means = (double*)malloc(MAX_STEPS * ACTION_DIM * sizeof(double));
    ep->rewards = (double*)malloc(MAX_STEPS * sizeof(double));
    ep->returns = (double*)malloc(MAX_STEPS * sizeof(double));
    ep->length = 0;
    return ep;
}

// Free episode storage
void free_episode(Episode* ep) {
    free(ep->states);
    free(ep->actions);
    free(ep->means);
    free(ep->rewards);
    free(ep->returns);
    free(ep);
}

// Initialize RL parameters
RLParams* create_rl_params() {
    RLParams* params = (RLParams*)malloc(sizeof(RLParams));
    params->learning_rate = 0.001;
    params->gamma = 0.99;
    params->sigma = 0.1;
    params->hidden_dim = HIDDEN_DIM;
    params->batch_size = MAX_STEPS;
    return params;
}

// Sample action from Gaussian policy
void sample_action(Net* policy, double* state, double* action, double* mean, 
                  double sigma) {
    // Create temporary storage for single state input
    float* state_input = (float*)malloc(STATE_DIM * sizeof(float));
    for(int i = 0; i < STATE_DIM; i++) {
        state_input[i] = (float)state[i];
    }
    
    // Forward pass through policy network
    forward_pass(policy, state_input);
    
    // Copy means and sample actions
    for(int i = 0; i < ACTION_DIM; i++) {
        mean[i] = policy->predictions[i];
        action[i] = mean[i] + sigma * randn();
    }
    
    free(state_input);
}

// Compute discounted returns
void compute_returns(Episode* ep, double gamma) {
    double cumulative = 0.0;
    for(int t = ep->length - 1; t >= 0; t--) {
        cumulative = ep->rewards[t] + gamma * cumulative;
        ep->returns[t] = cumulative;
    }
}

// Prepare training batch for policy update
void prepare_training_batch(Episode* ep, float* X, float* y, double sigma) {
    // Convert episode data to training batch
    for(int t = 0; t < ep->length; t++) {
        // Input state
        for(int i = 0; i < STATE_DIM; i++) {
            X[t * STATE_DIM + i] = (float)ep->states[t * STATE_DIM + i];
        }
        
        // Compute "targets" using the REINFORCE gradient formula
        // target = mean + (action - mean) * (return / sigma²)
        for(int i = 0; i < ACTION_DIM; i++) {
            double mean = ep->means[t * ACTION_DIM + i];
            double action = ep->actions[t * ACTION_DIM + i];
            double G = ep->returns[t];
            
            y[t * ACTION_DIM + i] = (float)(
                mean + (action - mean) * (G / (sigma * sigma))
            );
        }
    }
}

// Update policy network using REINFORCE
void update_policy(Net* policy, Episode* ep, RLParams* params) {
    // Compute returns
    compute_returns(ep, params->gamma);
    
    // Prepare training batch
    float* X = (float*)malloc(ep->length * STATE_DIM * sizeof(float));
    float* y = (float*)malloc(ep->length * ACTION_DIM * sizeof(float));
    
    prepare_training_batch(ep, X, y, params->sigma);
    
    // Update network
    zero_gradients(policy);
    forward_pass(policy, X);
    float loss = calculate_loss(policy, y);
    backward_pass(policy, X);
    update_weights(policy, params->learning_rate);
    
    free(X);
    free(y);
}

// Evaluate policy (without exploration noise)
double evaluate_policy(Net* policy, Environment* env, bool render) {
    state_from_quad(env);
    double total_reward = 0.0;
    double mean[ACTION_DIM];
    double action[ACTION_DIM];
    
    while(!is_terminal(env)) {
        // Get deterministic action (mean, no noise)
        sample_action(policy, env->state, action, mean, 0.0);
        
        // Take step
        double reward = environment_step(env, action);
        total_reward += reward;
        
        // Optional rendering
        if(render) {
            // TODO: Add visualization code here
            // This would interface with the existing raytracer
        }
    }
    
    return total_reward;
}

// Training statistics
typedef struct {
    double* episode_rewards;
    double* eval_rewards;
    int episode_count;
    int eval_interval;
    char log_file[256];
} TrainingStats;

// Initialize training statistics
TrainingStats* create_training_stats(int num_episodes, int eval_interval) {
    TrainingStats* stats = (TrainingStats*)malloc(sizeof(TrainingStats));
    stats->episode_rewards = (double*)malloc(num_episodes * sizeof(double));
    stats->eval_rewards = (double*)malloc((num_episodes/eval_interval + 1) * sizeof(double));
    stats->episode_count = num_episodes;
    stats->eval_interval = eval_interval;
    
    // Create timestamp for log file
    time_t now = time(NULL);
    strftime(stats->log_file, sizeof(stats->log_file), 
             "training_%Y%m%d_%H%M%S.csv", localtime(&now));
    return stats;
}

// Log training progress
void log_progress(TrainingStats* stats, int episode, double train_reward, 
                 double eval_reward) {
    static FILE* f = NULL;
    if (!f) {
        f = fopen(stats->log_file, "w");
        fprintf(f, "Episode,TrainReward,EvalReward\n");
    }
    fprintf(f, "%d,%f,%f\n", episode, train_reward, eval_reward);
    fflush(f);
}

// Main training loop
void train_policy(int num_episodes) {
    // Initialize environment and policy network
    Environment* env = create_environment();
    Net* policy = init_net(STATE_DIM, HIDDEN_DIM, ACTION_DIM, MAX_STEPS);
    RLParams* params = create_rl_params();
    Episode* ep = create_episode();
    TrainingStats* stats = create_training_stats(num_episodes, 100);
    
    // Training loop
    for(int episode = 0; episode < num_episodes; episode++) {
        // Reset environment
        environment_reset(env);
        ep->length = 0;
        double episode_reward = 0.0;
        
        // Episode loop
        while(!is_terminal(env)) {
            // Get current state
            int t = ep->length;
            memcpy(&ep->states[t * STATE_DIM], env->state, 
                   STATE_DIM * sizeof(double));
            
            // Sample action from policy
            sample_action(policy, env->state, 
                         &ep->actions[t * ACTION_DIM],
                         &ep->means[t * ACTION_DIM], 
                         params->sigma);
            
            // Take step in environment
            double reward = environment_step(env, &ep->actions[t * ACTION_DIM]);
            ep->rewards[t] = reward;
            episode_reward += reward;
            
            ep->length++;
        }
        
        // Update policy using REINFORCE
        update_policy(policy, ep, params);
        
        // Store training statistics
        stats->episode_rewards[episode] = episode_reward;
        
        // Periodic evaluation
        if((episode + 1) % stats->eval_interval == 0) {
            environment_reset(env);
            double eval_reward = evaluate_policy(policy, env, false);
            stats->eval_rewards[episode/stats->eval_interval] = eval_reward;
            
            // Log progress
            log_progress(stats, episode, episode_reward, eval_reward);
            
            printf("Episode %d/%d: Train reward = %.2f, Eval reward = %.2f\n",
                   episode + 1, num_episodes, episode_reward, eval_reward);
            
            // Save model checkpoint
            char model_file[256];
            sprintf(model_file, "policy_checkpoint_%d.bin", episode + 1);
            save_model(policy, model_file);
        }
    }
    
    // Final evaluation with rendering
    printf("\nFinal policy evaluation with rendering...\n");
    environment_reset(env);
    evaluate_policy(policy, env, true);
    
    // Cleanup
    free_environment(env);
    free_net(policy);
    free(params);
    free_episode(ep);
    free(stats->episode_rewards);
    free(stats->eval_rewards);
    free(stats);
}

// Visualization integration
void render_quad_state(Environment* env) {
    // TODO: Interface with raytracer code
    // This would update the drone mesh position/orientation
    // and render a new frame
}

int main(int argc, char** argv) {
    // Seed random number generator
    srand(time(NULL));
    
    // Parse command line arguments
    int num_episodes = 1000;
    if(argc > 1) {
        num_episodes = atoi(argv[1]);
    }
    
    // Run training
    train_policy(num_episodes);
    
    return 0;
}