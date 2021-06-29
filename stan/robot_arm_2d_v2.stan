data {
    int<lower=0> N;
    matrix[2, N] q;
    matrix[2, N] dq;
    matrix[2, N] tau;
    real<lower=0.0> g;
    real<lower=0.0> dt;
}
//transformed data {
//    row_vector[N] sq2 = sin(q[2, :]);
//    row_vector[N] cq1 = cos(q[1, :]);
//    row_vector[N] cq1q2 = cos(q[1, :] + q[2, :]);
//}
parameters {
    real <lower=1e-8> m1;
    real <lower=1e-8> m2;
    real <lower=1e-8> l1;
    real <lower=1e-8> l2;
    real <lower=0.0> d1;
    real <lower=0.0> d2;
    real <lower=1e-8> r;
}

transformed parameters {
    real theta1 = l2^2 * m2 + l1^2 * (m1 + m2);
    real theta2 = l1 * l2 * m2;
    real theta3 = l2^2 * m2;
    real theta4 = l2 * m2;
    real theta5 = l1 * (m1 + m2);

}

model {
    row_vector[N] m11 = theta1 + 2 * theta2 * cos(q[2, :]);
    row_vector[N] m12 = theta3 + theta2 * cos(q[2, :]);
    real m22 = theta3;
    row_vector[N] mdet = m11 * m22 - m12 .* m12;

    row_vector[N] u1 = g * theta4 * cos(q[1, :] + q[2, :]) + theta5 * g * cos(q[1, :]);
    row_vector[N] u2 = theta4 * g * cos(q[1, :] + q[2, :]);
    row_vector[N] c2 = -2 * theta2 * (dq[1, :] .* dq[1, :] .* sin(q[2, :])) - 2 * theta2 * dq[1, :] .* dq[2, :] .* sin(q[2, :]);

    row_vector[N] f1 = - u1 - d1 * dq[1] + tau[1, :];
    row_vector[N] f2 = c2 - u2 - d2 * dq[2] + tau[2, :];

    row_vector[N] ddq1hat = (m22 * f1 - m12 .* f2) ./ mdet;
    row_vector[N] ddq2hat = (m11 .* f2 - m12 .* f1) ./ mdet;

    row_vector[N-1] dq1hat = dq[1, 1:N-1] + dt * ddq1hat[1:N-1];
    row_vector[N-1] dq2hat = dq[2, 1:N-1] + dt * ddq2hat[1:N-1];

    r ~ cauchy(0, 1.0);

    dq[1, 2:N] ~ normal(dq1hat, r);
    dq[2, 2:N] ~ normal(dq2hat, r);

//    ddq[1, :] .* m11 + ddq[2, :] .* m12 - f1 ~ normal(0.0, r);
//    ddq[1,:] .* m12 + ddq[2, :] * m22 - f2 ~ normal(0.0, r);

}