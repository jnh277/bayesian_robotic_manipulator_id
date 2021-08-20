data {
    int<lower=0> N;
    matrix[2, N] q;
    matrix[2, N] dq;
    matrix[2, N] ddq;
    matrix[2, N] tau;
    real a1;
}

parameters {
    real<lower=1e-6> m_1;
    real<lower=1e-6> m_2;
    vector[3] r_1;
    vector[3] r_2;
    real<lower=0.0> fv_1;
    real<lower=0.0> fv_2;
    real<lower=1e-6> r;     // measurement standard deviation

    // parameters for first inertia
    real <lower=0.0> eig12;        // eigen values
    real <lower=0.0> eig13;
    real <lower=fmax(eig13-eig12,eig12-eig13), upper=eig12+eig13> eig11;
    unit_vector[4] quat1;

    // parameters for second inertia
    real <lower=0.0> eig22;        // eigen values
    real <lower=0.0> eig23;
    real <lower=fmax(eig23-eig22,eig22-eig23), upper=eig22+eig23> eig21;
    unit_vector[4] quat2;
}

transformed parameters {
    // need l_1, l_2, L_1, L_2

    // transformed parameters for first inertia
    vector[3] eigs1;
    matrix[3,3] R1;
    matrix[3,3] I_1;
    R1[1,1] = quat1[1]*quat1[1] + quat1[2]*quat1[2] - quat1[3]*quat1[3] - quat1[4]*quat1[4]; R1[1,2] = 2*(quat1[2]*quat1[3]-quat1[1]*quat1[4]); R1[1,3] = 2*(quat1[1]*quat1[3]+quat1[2]*quat1[4]);
    R1[2,1] = 2*(quat1[2]*quat1[3]+quat1[1]*quat1[4]); R1[2,2] = quat1[1]*quat1[1] - quat1[2]*quat1[2] + quat1[3]*quat1[3] - quat1[4]*quat1[4]; R1[2,3] = 2*(quat1[3]*quat1[4]-quat1[1]*quat1[2]);
    R1[3,1] = 2*(quat1[2]*quat1[4]-quat1[1]*quat1[3]); R1[3,2] = 2*(quat1[1]*quat1[2]+quat1[3]*quat1[4]); R1[3,3] = quat1[1]*quat1[1] - quat1[2]*quat1[2] - quat1[3]*quat1[3] + quat1[4]*quat1[4];
    eigs1[1] = eig11; eigs1[2] = eig12; eigs1[3] = eig13;
    I_1 = diag_post_multiply(R1,eigs1) * R1';

    // transformed parameters for second inertia
    vector[3] eigs2;
    matrix[3,3] R2;
    matrix[3,3] I_2;
    R2[1,1] = quat2[1]*quat2[1] + quat2[2]*quat2[2] - quat2[3]*quat2[3] - quat2[4]*quat2[4]; R2[1,2] = 2*(quat2[2]*quat2[3]-quat2[1]*quat2[4]); R2[1,3] = 2*(quat2[1]*quat2[3]+quat2[2]*quat2[4]);
    R2[2,1] = 2*(quat2[2]*quat2[3]+quat2[1]*quat2[4]); R2[2,2] = quat2[1]*quat2[1] - quat2[2]*quat2[2] + quat2[3]*quat2[3] - quat2[4]*quat2[4]; R2[2,3] = 2*(quat2[3]*quat2[4]-quat2[1]*quat2[2]);
    R2[3,1] = 2*(quat2[2]*quat2[4]-quat2[1]*quat2[3]); R2[3,2] = 2*(quat2[1]*quat2[2]+quat2[3]*quat2[4]); R2[3,3] = quat2[1]*quat2[1] - quat2[2]*quat2[2] - quat2[3]*quat2[3] + quat2[4]*quat2[4];
    eigs2[1] = eig21; eigs2[2] = eig22; eigs2[3] = eig23;
    I_2 = diag_post_multiply(R2,eigs2) * R2';


    real L_1xx, L_1yy, L_1zz, L_1xy, L_1xz, L_1yz;
    real L_2xx, L_2yy, L_2zz, L_2xy, L_2xz, L_2yz;
    vector[3] l_1 = r_1 * m_1;
    vector[3] l_2 = r_2 * m_2;


    L_1xx = I_1[1,1] + m_1*pow(r_1[2],2) + m_1*pow(r_1[3],2);
    L_1yy = I_1[2,2] + m_1*pow(r_1[1],2) + m_1*pow(r_1[3],2);
    L_1zz = I_1[3,3] + m_1*pow(r_1[1],2) + m_1*pow(r_1[2],2);
    L_1xy = I_1[1,2] - m_1*r_1[1]*r_1[2];
    L_1xz = I_1[1,3] - m_1*r_1[1]*r_1[3];
    L_1yz = I_1[2,3] - m_1*r_1[2]*r_1[3];

    L_2xx = I_2[1,1] + m_2*pow(r_2[2],2) + m_2*pow(r_2[3],2);
    L_2yy = I_2[2,2] + m_2*pow(r_2[1],2) + m_2*pow(r_2[3],2);
    L_2zz = I_2[3,3] + m_2*pow(r_2[1],2) + m_2*pow(r_2[2],2);
    L_2xy = I_2[1,2] - m_2*r_2[1]*r_2[2];
    L_2xz = I_2[1,3] - m_2*r_2[1]*r_2[3];
    L_2yz = I_2[2,3] - m_2*r_2[2]*r_2[3];

    row_vector[22] params = [L_1xx, L_1xy, L_1xz, L_1yy, L_1yz, L_1zz, l_1[1], l_1[2], l_1[3], m_1, fv_1,
                        L_2xx, L_2xy, L_2xz, L_2yy, L_2yz, L_2zz, l_2[1], l_2[2], l_2[3], m_2, fv_2];



}

model {
    matrix[2, N] tau_hat;
    row_vector[N] x0 = cos(q[2, :]);
    row_vector[N] x1 = ddq[1, :] + ddq[2, :];
    row_vector[N] x2 = -((dq[1, :] + dq[2, :]) .* (dq[1, :] + dq[2, :]));
    row_vector[N] x3 = 9.81 * sin(q[1, :]);
    row_vector[N] x4 = -a1*dq[1, :] .* dq[1,:] + x3;
    row_vector[N] x5 = sin(q[2, :]);
    row_vector[N] x6 = 9.81*cos(q[1, :]);
    row_vector[N] x7 = a1*ddq[1, :] + x6;
    row_vector[N] x8 = x0 .* x7 - x4 .* x5;
    row_vector[N] x9 = x0 .* x4 + x5 .* x7;
    row_vector[N] x10 = params[17] * x1 + params[18] * x8 - params[19] * x9;
    //
    tau_hat[1, :] = a1*(x0 .* (params[18]*x1 + params[19]*x2 + params[21]*x8) + x5 .* (params[18]*x2 - params[19]*x1 + params[21]*x9)) + ddq[1, :]*params[6] + dq[1, :]*params[11] + params[7]*x6 - params[8]*x3 + x10;
    tau_hat[2, :] = dq[2, :]*params[22] + x10;

    // priors
    r ~ cauchy(0, 1.0);
    to_vector(I_1) ~ cauchy(0, 1.0);
    to_vector(I_2) ~ cauchy(0, 1.0);
    r_1 ~ cauchy(0, 1.0);
    r_2 ~ cauchy(0, 1.0);
    fv_1 ~ cauchy(0, 1.0);
    fv_2 ~ cauchy(0, 1.0);

    //
    tau[1, :] ~ normal(tau_hat[1, :], r);
    tau[2, :] ~ normal(tau_hat[2, :], r);

}