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

void so3vee(const double* m, double* result) {
    result[0] = m[7];
    result[1] = m[2];
    result[2] = m[3];
}

// Matrix arithmetic
void addMat3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 9; i++) result[i] = a[i] + b[i];
}

void subMat3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 9; i++) result[i] = a[i] - b[i];
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

void normVec3f(const double* v, double* result) {
    double mag = sqrtf(dotVec3f(v, v));
    for(int i = 0; i < 3; i++) result[i] = v[i]/mag;
}

void inv4Mat4f(const double* m, double* result) {
    double s0 = m[0]*m[5] - m[4]*m[1];
    double s1 = m[0]*m[6] - m[4]*m[2];
    double s2 = m[0]*m[7] - m[4]*m[3];
    double s3 = m[1]*m[6] - m[5]*m[2];
    double s4 = m[1]*m[7] - m[5]*m[3];
    double s5 = m[2]*m[7] - m[6]*m[3];

    double c5 = m[10]*m[15] - m[14]*m[11];
    double c4 = m[9]*m[15] - m[13]*m[11];
    double c3 = m[9]*m[14] - m[13]*m[10];
    double c2 = m[8]*m[15] - m[12]*m[11];
    double c1 = m[8]*m[14] - m[12]*m[10];
    double c0 = m[8]*m[13] - m[12]*m[9];

    double det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
    
    if (det == 0.0) {
        // Handle error case
        return;
    }

    double invdet = 1.0/det;

    result[0] = (m[5]*c5 - m[6]*c4 + m[7]*c3)*invdet;
    result[1] = (-m[1]*c5 + m[2]*c4 - m[3]*c3)*invdet;
    result[2] = (m[13]*s5 - m[14]*s4 + m[15]*s3)*invdet;
    result[3] = (-m[9]*s5 + m[10]*s4 - m[11]*s3)*invdet;

    result[4] = (-m[4]*c5 + m[6]*c2 - m[7]*c1)*invdet;
    result[5] = (m[0]*c5 - m[2]*c2 + m[3]*c1)*invdet;
    result[6] = (-m[12]*s5 + m[14]*s2 - m[15]*s1)*invdet;
    result[7] = (m[8]*s5 - m[10]*s2 + m[11]*s1)*invdet;

    result[8] = (m[4]*c4 - m[5]*c2 + m[7]*c0)*invdet;
    result[9] = (-m[0]*c4 + m[1]*c2 - m[3]*c0)*invdet;
    result[10] = (m[12]*s4 - m[13]*s2 + m[15]*s0)*invdet;
    result[11] = (-m[8]*s4 + m[9]*s2 - m[11]*s0)*invdet;

    result[12] = (-m[4]*c3 + m[5]*c1 - m[6]*c0)*invdet;
    result[13] = (m[0]*c3 - m[1]*c1 + m[2]*c0)*invdet;
    result[14] = (-m[12]*s3 + m[13]*s1 - m[14]*s0)*invdet;
    result[15] = (m[8]*s3 - m[9]*s1 + m[10]*s0)*invdet;
}

void multMatVec4f(const double* m, const double* v, double* result) {
    result[0] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2] + m[3]*v[3];
    result[1] = m[4]*v[0] + m[5]*v[1] + m[6]*v[2] + m[7]*v[3];
    result[2] = m[8]*v[0] + m[9]*v[1] + m[10]*v[2] + m[11]*v[3];
    result[3] = m[12]*v[0] + m[13]*v[1] + m[14]*v[2] + m[15]*v[3];
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

#define K_P 0.2
#define K_V 0.6
#define K_R 0.6
#define K_W 0.6

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

void reset_quad(Quad* q, double x, double y, double z) {
    memcpy(q->omega, (double[]){0.0, 0.0, 0.0, 0.0}, 4 * sizeof(double));
    memcpy(q->linear_position_W, (double[]){x, y, z}, 3 * sizeof(double));
    memcpy(q->linear_velocity_W, (double[]){0.0, 0.0, 0.0}, 3 * sizeof(double));
    memcpy(q->angular_velocity_B, (double[]){0.0, 0.0, 0.0}, 3 * sizeof(double));
    memcpy(q->R_W_B, (double[]){1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, 9 * sizeof(double));
    memcpy(q->inertia, (double[]){0.01, 0.02, 0.01}, 3 * sizeof(double));
    memcpy(q->omega_next, (double[]){0.0, 0.0, 0.0, 0.0}, 4 * sizeof(double));
    memcpy(q->linear_acceleration_B_s, (double[]){0.0, 0.0, 0.0}, 3 * sizeof(double));
    memcpy(q->angular_velocity_B_s, (double[]){0.0, 0.0, 0.0}, 3 * sizeof(double));
    for(int i = 0; i < 3; i++) {
        q->accel_bias[i] = (2.0*((double)rand()/RAND_MAX) - 1.0) * ACCEL_BIAS;
        q->gyro_bias[i] = (2.0*((double)rand()/RAND_MAX) - 1.0) * GYRO_BIAS;
    }
}

Quad* init_quad(double x, double y, double z) {
    Quad* quad = malloc(sizeof(Quad));
    reset_quad(quad, x, y, z);
    return quad;
}

void print_quad(Quad* q) {
    printf("\rP: %.2f %.2f %.2f | R: %.2f %.2f %.2f %.2f", q->linear_position_W[0], q->linear_position_W[1], q->linear_position_W[2], q->omega[0], q->omega[1], q->omega[2], q->omega[3]);
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
    for(int i = 0; i < 4; i++) {
        double f_vector[3] = {0, f[i], 0};
        double tau_thrust[3];
        crossVec3f((double [4][3]){{-L, 0,  L}, { L, 0,  L}, { L, 0, -L}, {-L, 0, -L}}[i], f_vector, tau_thrust);
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
    transpMat3f(q->R_W_B, R_B_W);
    multMatVec3f(R_B_W, linear_acceleration_W, linear_acceleration_B);
    double gravity_B[3];
    multMatVec3f(R_B_W, (double[3]){0, GRAVITY, 0}, gravity_B);
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

void control_quad(Quad* q, double* control_input) {
    // 1. Calculate position and velocity errors
    double error_p[3], error_v[3];
    subVec3f(q->linear_position_W, (double[]){control_input[0], control_input[1], control_input[2]}, error_p);
    subVec3f(q->linear_velocity_W, (double[]){control_input[3], control_input[4], control_input[5]}, error_v);

    // 2. Calculate desired force vector in world frame
    double z_W_d[3], temp[3];
    multScalVec3f(-K_P, error_p, z_W_d);
    multScalVec3f(-K_V, error_v, temp);
    addVec3f(z_W_d, temp, z_W_d);
    
    // Add gravity compensation and desired acceleration
    double gravity_term[3] = {0, MASS * GRAVITY, 0};
    addVec3f(z_W_d, gravity_term, z_W_d);
    
    double accel_term[3];
    multScalVec3f(MASS, (double[]){0.0, 0.0, 0.0}, accel_term);
    addVec3f(z_W_d, accel_term, z_W_d);

    // 3. Calculate thrust magnitude
    double z_W_B[3];
    double y_body[3] = {0, 1, 0};
    multMatVec3f(q->R_W_B, y_body, z_W_B);
    control_input[0] = dotVec3f(z_W_d, z_W_B);

    // 4. Calculate desired rotation matrix
    double x_tilde_d_W[3] = {sin(control_input[6]), 0.0, cos(control_input[6])};
    double temp_cross1[3], temp_cross2[3];
    double R_W_d_column_0[3], R_W_d_column_1[3], R_W_d_column_2[3];
    
    crossVec3f(z_W_d, x_tilde_d_W, temp_cross1);
    crossVec3f(temp_cross1, z_W_d, temp_cross2);
    normVec3f(temp_cross2, R_W_d_column_0);
    normVec3f(temp_cross1, R_W_d_column_1);
    normVec3f(z_W_d, R_W_d_column_2);

    double R_W_d[9] = {
        R_W_d_column_1[0], R_W_d_column_2[0], R_W_d_column_0[0],
        R_W_d_column_1[1], R_W_d_column_2[1], R_W_d_column_0[1],
        R_W_d_column_1[2], R_W_d_column_2[2], R_W_d_column_0[2]
    };

    // 5. Calculate rotation error
    double R_W_d_T[9], R_W_B_T[9], temp_mat1[9], temp_mat2[9], temp_mat3[9];
    transpMat3f(R_W_d, R_W_d_T);
    transpMat3f(q->R_W_B, R_W_B_T);

    multMat3f(R_W_d_T, q->R_W_B, temp_mat1);
    multMat3f(R_W_B_T, R_W_d, temp_mat2);
    subMat3f(temp_mat1, temp_mat2, temp_mat3);

    double error_r[3];
    so3vee(temp_mat3, error_r);
    multScalVec3f(0.5, error_r, error_r);

    // 6. Calculate angular velocity error
    double temp_vec[3], error_w[3];
    multMat3f(R_W_d_T, q->R_W_B, temp_mat1);
    multMatVec3f(temp_mat1, (double[]){0.0, 0.0, 0.0}, temp_vec);
    subVec3f(q->angular_velocity_B, temp_vec, error_w);

    // 7. Calculate control torque
    double tau_B_control[3], temp_vec2[3];
    multScalVec3f(-K_R, error_r, tau_B_control);
    multScalVec3f(-K_W, error_w, temp_vec2);
    addVec3f(tau_B_control, temp_vec2, tau_B_control);

    // Add angular momentum terms
    double I_mat[9], temp_vec3[3], temp_vec4[3];
    vecToDiagMat3f(q->inertia, I_mat);
    multMatVec3f(I_mat, q->angular_velocity_B, temp_vec3);
    crossVec3f(q->angular_velocity_B, temp_vec3, temp_vec4);
    addVec3f(tau_B_control, temp_vec4, tau_B_control);

    // Add feedforward terms
    double term_0[3], term_1[3], temp_vec5[3];
    multMatVec3f(R_W_d, (double[]){0.0, 0.0, 0.0}, temp_vec);
    multMatVec3f(R_W_B_T, temp_vec, term_0);

    multMatVec3f(R_W_d, (double[]){0.0, 0.0, 0.0}, temp_vec);
    multMatVec3f(R_W_B_T, temp_vec, temp_vec2);
    crossVec3f((double[]){0.0, 0.0, 0.0}, temp_vec2, term_1);

    subVec3f(term_1, term_0, temp_vec5);
    multMatVec3f(I_mat, temp_vec5, temp_vec);
    subVec3f(tau_B_control, temp_vec, tau_B_control);

    // 8. Calculate rotor speeds
    double F_bar[16] = {
        K_F, K_F, K_F, K_F,   // Thrust coefficients
        0, 0, 0, 0,           // Roll moments
        K_M, -K_M, K_M, -K_M, // Yaw moments
        0, 0, 0, 0            // Pitch moments
    };

    // Calculate roll and pitch moments
    for(int i = 0; i < 4; i++) {
        double moment[3];
        double pos_scaled[3];
        multScalVec3f(K_F, (double [4][3]){{-L, 0,  L}, { L, 0,  L}, { L, 0, -L}, {-L, 0, -L}}[i], pos_scaled);
        crossVec3f(pos_scaled, (double[3]){0, 1, 0}, moment);
        F_bar[4 + i]  = moment[0];  // Roll
        F_bar[12 + i] = moment[2];  // Pitch
    }

    // 9. Calculate and update rotor speeds
    double F_bar_inv[16];
    inv4Mat4f(F_bar, F_bar_inv);
    double omega_sign_square[4];
    multMatVec4f(F_bar_inv, (double[]){control_input[0], tau_B_control[0], tau_B_control[1], tau_B_control[2]}, omega_sign_square);

    for(int i = 0; i < 4; i++) {
        q->omega_next[i] = sqrt(fabs(omega_sign_square[i]));
    }
}

#endif // QUAD_H