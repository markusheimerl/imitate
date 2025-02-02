#ifndef QUAD_H
#define QUAD_H

#include <math.h>

// 3x3 Matrix Operations
void multMat3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            result[i*3 + j] = a[i*3]*b[j] + a[i*3+1]*b[j+3] + a[i*3+2]*b[j+6];
}

void multMatVec3f(const double* m, const double* v, double* result) {
    result[0] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2];
    result[1] = m[3]*v[0] + m[4]*v[1] + m[5]*v[2];
    result[2] = m[6]*v[0] + m[7]*v[1] + m[8]*v[2];
}

void vecToDiagMat3f(const double* v, double* result) {
    for(int i = 0; i < 9; i++) result[i] = 0;
    result[0] = v[0];
    result[4] = v[1];
    result[8] = v[2];
}

void transpMat3f(const double* m, double* result) {
    result[0] = m[0]; result[1] = m[3]; result[2] = m[6];
    result[3] = m[1]; result[4] = m[4]; result[5] = m[7];
    result[6] = m[2]; result[7] = m[5]; result[8] = m[8];
}

void so3hat(const double* v, double* result) {
    result[0]=0; result[1]=-v[2]; result[2]=v[1];
    result[3]=v[2]; result[4]=0; result[5]=-v[0];
    result[6]=-v[1]; result[7]=v[0]; result[8]=0;
}

// Matrix arithmetic
void addMat3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 9; i++) result[i] = a[i] + b[i];
}

void multScalMat3f(double s, const double* m, double* result) {
    for(int i = 0; i < 9; i++) result[i] = s * m[i];
}

// Vector Operations
void crossVec3f(const double* a, const double* b, double* result) {
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
}

void multScalVec3f(double s, const double* v, double* result) {
    for(int i = 0; i < 3; i++) result[i] = s * v[i];
}

void addVec3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 3; i++) result[i] = a[i] + b[i];
}

void subVec3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 3; i++) result[i] = a[i] - b[i];
}

double dotVec3f(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void orthonormalize_rotation_matrix(double* R) {
    double x[3], y[3], z[3];
    double temp[3];
    
    // Extract columns
    for(int i = 0; i < 3; i++) {
        x[i] = R[i];      // First column
        y[i] = R[i + 3];  // Second column
        z[i] = R[i + 6];  // Third column
    }
    
    // Normalize x
    double norm_x = sqrt(dotVec3f(x, x));
    multScalVec3f(1.0/norm_x, x, x);
    
    // Make y orthogonal to x
    double dot_xy = dotVec3f(x, y);
    multScalVec3f(dot_xy, x, temp);
    subVec3f(y, temp, y);
    // Normalize y
    double norm_y = sqrt(dotVec3f(y, y));
    multScalVec3f(1.0/norm_y, y, y);
    
    // Make z orthogonal to x and y using cross product
    crossVec3f(x, y, z);
    // z is automatically normalized since x and y are orthonormal
    
    // Put back into matrix
    for(int i = 0; i < 3; i++) {
        R[i] = x[i];      // First column
        R[i + 3] = y[i];  // Second column
        R[i + 6] = z[i];  // Third column
    }
}

// Constants
#define K_F 0.0004905
#define K_M 0.00004905
#define L 0.25
#define L_SQRT2 (L / sqrtf(2.0))
#define GRAVITY 9.81
#define MASS 0.5
#define OMEGA_MIN 30.0
#define OMEGA_MAX 70.0

#define ACCEL_NOISE_STDDEV 0.1
#define GYRO_NOISE_STDDEV 0.01
#define ACCEL_BIAS 0.05
#define GYRO_BIAS 0.005

typedef struct {
    // State variables
    double omega[4];
    double linear_position_W[3];
    double linear_velocity_W[3];
    double angular_velocity_B[3];
    double R_W_B[9];
    double inertia[3];
    double omega_next[4];

    // Sensor variables
    double linear_acceleration_B_s[3]; // Accelerometer
    double angular_velocity_B_s[3]; // Gyroscope
    double accel_bias[3];
    double gyro_bias[3];
} Quad;

Quad create_quad(double x, double y, double z) {
    Quad q;
    static const double zero4[4] = {0.0, 0.0, 0.0, 0.0};
    static const double zero3[3] = {0.0, 0.0, 0.0};
    static const double identity9[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    static const double inertia3[3] = {0.01, 0.02, 0.01};
    double pos3[3] = {x, y, z};

    memcpy(q.omega, zero4, 4 * sizeof(double));
    memcpy(q.linear_position_W, pos3, 3 * sizeof(double));
    memcpy(q.linear_velocity_W, zero3, 3 * sizeof(double));
    memcpy(q.angular_velocity_B, zero3, 3 * sizeof(double));
    memcpy(q.R_W_B, identity9, 9 * sizeof(double));
    memcpy(q.inertia, inertia3, 3 * sizeof(double));
    memcpy(q.omega_next, zero4, 4 * sizeof(double));
    memcpy(q.linear_acceleration_B_s, zero3, 3 * sizeof(double));
    memcpy(q.angular_velocity_B_s, zero3, 3 * sizeof(double));
    
    for(int i = 0; i < 3; i++) {
        q.accel_bias[i] = (2.0*((double)rand()/RAND_MAX) - 1.0) * ACCEL_BIAS;
        q.gyro_bias[i] = (2.0*((double)rand()/RAND_MAX) - 1.0) * GYRO_BIAS;
    }
    return q;
}

void get_quad_state(Quad q, double* state) {
    memcpy(state, q.linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q.linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q.angular_velocity_B, 3 * sizeof(double));
    state[9] = q.R_W_B[0];
    state[10] = q.R_W_B[4];
    state[11] = q.R_W_B[8];
}

static double gaussian_noise(double stddev) {
    double u1 = (double)rand() / RAND_MAX, u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) * stddev;
}

void update_quad(Quad* q, double dt) {
    // 1. Declare arrays and calculate rotor forces/moments
    double f[4], m[4];
    for(int i = 0; i < 4; i++) {
        q->omega[i] = fmax(fmin(q->omega[i], OMEGA_MAX), OMEGA_MIN);
        double omega_sq = q->omega[i] * fabs(q->omega[i]);
        f[i] = K_F * omega_sq;
        m[i] = K_M * omega_sq;
    }

    // 2. Calculate total thrust force in body frame (only y component is non-zero)
    double f_B_thrust[3] = {0, f[0] + f[1] + f[2] + f[3], 0};

    // 3. Initialize with drag torque (only y component is non-zero)
    double tau_B[3] = {0, m[0] - m[1] + m[2] - m[3], 0};

    // 4. Add thrust torques
    static const double rotor_positions[4][3] = {
        {-L, 0,  L},
        { L, 0,  L},
        { L, 0, -L},
        {-L, 0, -L}
    };

    for(int i = 0; i < 4; i++) {
        double f_vector[3] = {0, f[i], 0};
        double tau_thrust[3];
        crossVec3f(rotor_positions[i], f_vector, tau_thrust);
        addVec3f(tau_B, tau_thrust, tau_B);
    }

    // 5. Transform thrust to world frame and calculate linear acceleration
    double f_thrust_W[3];
    multMatVec3f(q->R_W_B, f_B_thrust, f_thrust_W);
    
    double linear_acceleration_W[3];
    for(int i = 0; i < 3; i++) {
        linear_acceleration_W[i] = f_thrust_W[i] / MASS;
    }
    linear_acceleration_W[1] -= GRAVITY;  // Add gravity

    // 6. Calculate angular acceleration
    double I_mat[9];
    vecToDiagMat3f(q->inertia, I_mat);
    
    double h_B[3];
    multMatVec3f(I_mat, q->angular_velocity_B, h_B);

    double w_cross_h[3];
    crossVec3f(q->angular_velocity_B, h_B, w_cross_h);

    double angular_acceleration_B[3];
    for(int i = 0; i < 3; i++) {
        angular_acceleration_B[i] = (-w_cross_h[i] + tau_B[i]) / q->inertia[i];
    }

    // 7. Update states with Euler integration
    for(int i = 0; i < 3; i++) {
        q->linear_velocity_W[i] += dt * linear_acceleration_W[i];
        q->linear_position_W[i] += dt * q->linear_velocity_W[i];
        q->angular_velocity_B[i] += dt * angular_acceleration_B[i];
    }

    // Ensure the quadcopter doesn't go below ground level
    if (q->linear_position_W[1] < 0.0) q->linear_position_W[1] = 0.0;

    // 8. Update rotation matrix
    double w_hat[9];
    so3hat(q->angular_velocity_B, w_hat);

    double R_dot[9];
    multMat3f(q->R_W_B, w_hat, R_dot);

    double R_dot_scaled[9];
    multScalMat3f(dt, R_dot, R_dot_scaled);

    double R_new[9];
    addMat3f(q->R_W_B, R_dot_scaled, R_new);
    for (int i = 0; i < 9; i++) q->R_W_B[i] = R_new[i];

    // 9. Ensure rotation matrix stays orthonormal
    orthonormalize_rotation_matrix(q->R_W_B);

    // 10. Calculate sensor readings with noise
    double linear_acceleration_B[3], R_B_W[9];
    static const double gravity_vec[3] = {0, GRAVITY, 0};
    
    transpMat3f(q->R_W_B, R_B_W);
    multMatVec3f(R_B_W, linear_acceleration_W, linear_acceleration_B);
    double gravity_B[3];
    multMatVec3f(R_B_W, gravity_vec, gravity_B);
    subVec3f(linear_acceleration_B, gravity_B, linear_acceleration_B);
    
    for(int i = 0; i < 3; i++) {
        q->linear_acceleration_B_s[i] = linear_acceleration_B[i] + gaussian_noise(ACCEL_NOISE_STDDEV) + q->accel_bias[i];
        q->angular_velocity_B_s[i] = q->angular_velocity_B[i] + gaussian_noise(GYRO_NOISE_STDDEV) + q->gyro_bias[i];
    }

    // 11. Update rotor speeds
    for(int i = 0; i < 4; i++) {
        q->omega[i] = fmax(OMEGA_MIN, fmin(OMEGA_MAX, q->omega_next[i]));
    }
}

#endif // QUAD_H