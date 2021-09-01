data {
    int<lower=0> N;
    matrix[3, N] q;
    matrix[3, N] dq;
    matrix[3, N] ddq;
    matrix[3, N] tau;
    real a1;
    real d0;
}

parameters {
	real<lower=1e-6> r;
	real<lower=1e-6> m_1;
	vector[3] r_1;
	real<lower=0.0> fv_1;
	cov_matrix[3] I_1;
//	real <lower=0.0> eig12;
//	real <lower=0.0> eig13;
//	real <lower=fmax(eig13-eig12,eig12-eig13), upper=eig12+eig13> eig11;
//	unit_vector[4] quat1;
	real<lower=1e-6> m_2;
	vector[3] r_2;
	real<lower=0.0> fv_2;
	cov_matrix[3] I_2;
//	real <lower=0.0> eig22;
//	real <lower=0.0> eig23;
//	real <lower=fmax(eig23-eig22,eig22-eig23), upper=eig22+eig23> eig21;
//	unit_vector[4] quat2;
	real<lower=1e-6> m_3;
	vector[3] r_3;
	real<lower=0.0> fv_3;
	cov_matrix[3] I_3;
//	real <lower=0.0> eig32;
//	real <lower=0.0> eig33;
//	real <lower=fmax(eig33-eig32,eig32-eig33), upper=eig32+eig33> eig31;
//	unit_vector[4] quat3;
}

transformed parameters {
//	vector[3] eigs1;
//	matrix[3,3] R1;
//	matrix[3,3] I_1;
//	R1[1,1] = quat1[1]*quat1[1] + quat1[2]*quat1[2] - quat1[3]*quat1[3] - quat1[4]*quat1[4]; R1[1,2] = 2*(quat1[2]*quat1[3]-quat1[1]*quat1[4]); R1[1,3] = 2*(quat1[1]*quat1[3]+quat1[2]*quat1[4]);
//	R1[2,1] = 2*(quat1[2]*quat1[3]+quat1[1]*quat1[4]); R1[2,2] = quat1[1]*quat1[1] - quat1[2]*quat1[2] + quat1[3]*quat1[3] - quat1[4]*quat1[4]; R1[2,3] = 2*(quat1[3]*quat1[4]-quat1[1]*quat1[2]);
//	R1[3,1] = 2*(quat1[2]*quat1[4]-quat1[1]*quat1[3]); R1[3,2] = 2*(quat1[1]*quat1[2]+quat1[3]*quat1[4]); R1[3,3] = quat1[1]*quat1[1] - quat1[2]*quat1[2] - quat1[3]*quat1[3] + quat1[4]*quat1[4];
//	eigs1[1] = eig11; eigs1[2] = eig12; eigs1[3] = eig13;
//	I_1 = diag_post_multiply(R1,eigs1) * R1';
	real L_1xx, L_1yy, L_1zz, L_1xy, L_1xz, L_1yz;
	vector[3] l_1 = r_1 * m_1;
	L_1xx = I_1[1,1] + m_1*pow(r_1[2],2) + m_1*pow(r_1[3],2);
	L_1yy = I_1[2,2] + m_1*pow(r_1[1],2) + m_1*pow(r_1[3],2);
	L_1zz = I_1[3,3] + m_1*pow(r_1[1],2) + m_1*pow(r_1[2],2);
	L_1xy = I_1[1,2] - m_1*r_1[1]*r_1[2];
	L_1xz = I_1[1,3] - m_1*r_1[1]*r_1[3];
	L_1yz = I_1[2,3] - m_1*r_1[2]*r_1[3];
//	vector[3] eigs2;
//	matrix[3,3] R2;
//	matrix[3,3] I_2;
//	R2[1,1] = quat2[1]*quat2[1] + quat2[2]*quat2[2] - quat2[3]*quat2[3] - quat2[4]*quat2[4]; R2[1,2] = 2*(quat2[2]*quat2[3]-quat2[1]*quat2[4]); R2[1,3] = 2*(quat2[1]*quat2[3]+quat2[2]*quat2[4]);
//	R2[2,1] = 2*(quat2[2]*quat2[3]+quat2[1]*quat2[4]); R2[2,2] = quat2[1]*quat2[1] - quat2[2]*quat2[2] + quat2[3]*quat2[3] - quat2[4]*quat2[4]; R2[2,3] = 2*(quat2[3]*quat2[4]-quat2[1]*quat2[2]);
//	R2[3,1] = 2*(quat2[2]*quat2[4]-quat2[1]*quat2[3]); R2[3,2] = 2*(quat2[1]*quat2[2]+quat2[3]*quat2[4]); R2[3,3] = quat2[1]*quat2[1] - quat2[2]*quat2[2] - quat2[3]*quat2[3] + quat2[4]*quat2[4];
//	eigs2[1] = eig21; eigs2[2] = eig22; eigs2[3] = eig23;
//	I_2 = diag_post_multiply(R2,eigs2) * R2';
	real L_2xx, L_2yy, L_2zz, L_2xy, L_2xz, L_2yz;
	vector[3] l_2 = r_2 * m_2;
	L_2xx = I_2[1,1] + m_2*pow(r_2[2],2) + m_2*pow(r_2[3],2);
	L_2yy = I_2[2,2] + m_2*pow(r_2[1],2) + m_2*pow(r_2[3],2);
	L_2zz = I_2[3,3] + m_2*pow(r_2[1],2) + m_2*pow(r_2[2],2);
	L_2xy = I_2[1,2] - m_2*r_2[1]*r_2[2];
	L_2xz = I_2[1,3] - m_2*r_2[1]*r_2[3];
	L_2yz = I_2[2,3] - m_2*r_2[2]*r_2[3];
//	vector[3] eigs3;
//	matrix[3,3] R3;
//	matrix[3,3] I_3;
//	R3[1,1] = quat3[1]*quat3[1] + quat3[2]*quat3[2] - quat3[3]*quat3[3] - quat3[4]*quat3[4]; R3[1,2] = 2*(quat3[2]*quat3[3]-quat3[1]*quat3[4]); R3[1,3] = 2*(quat3[1]*quat3[3]+quat3[2]*quat3[4]);
//	R3[2,1] = 2*(quat3[2]*quat3[3]+quat3[1]*quat3[4]); R3[2,2] = quat3[1]*quat3[1] - quat3[2]*quat3[2] + quat3[3]*quat3[3] - quat3[4]*quat3[4]; R3[2,3] = 2*(quat3[3]*quat3[4]-quat3[1]*quat3[2]);
//	R3[3,1] = 2*(quat3[2]*quat3[4]-quat3[1]*quat3[3]); R3[3,2] = 2*(quat3[1]*quat3[2]+quat3[3]*quat3[4]); R3[3,3] = quat3[1]*quat3[1] - quat3[2]*quat3[2] - quat3[3]*quat3[3] + quat3[4]*quat3[4];
//	eigs3[1] = eig31; eigs3[2] = eig32; eigs3[3] = eig33;
//	I_3 = diag_post_multiply(R3,eigs3) * R3';
	real L_3xx, L_3yy, L_3zz, L_3xy, L_3xz, L_3yz;
	vector[3] l_3 = r_3 * m_3;
	L_3xx = I_3[1,1] + m_3*pow(r_3[2],2) + m_3*pow(r_3[3],2);
	L_3yy = I_3[2,2] + m_3*pow(r_3[1],2) + m_3*pow(r_3[3],2);
	L_3zz = I_3[3,3] + m_3*pow(r_3[1],2) + m_3*pow(r_3[2],2);
	L_3xy = I_3[1,2] - m_3*r_3[1]*r_3[2];
	L_3xz = I_3[1,3] - m_3*r_3[1]*r_3[3];
	L_3yz = I_3[2,3] - m_3*r_3[2]*r_3[3];
	row_vector[22] params = [L_1xx, L_1xy, L_1xz, L_1yy, L_1yz, L_1zz, l_1[1], l_1[2], l_1[3], m_1, fv_1,
				L_2xx, L_2xy, L_2xz, L_2yy, L_2yz, L_2zz, l_2[1], l_2[2], l_2[3], m_2, fv_2,
				L_3xx, L_3xy, L_3xz, L_3yy, L_3yz, L_3zz, l_3[1], l_3[2], l_3[3], m_3, fv_3];
}
model {
	matrix[3,N] tau_hat;
  row_vector[N] x0 = sin(q[3,:]);
	row_vector[N] x1 = sin(q[2,:]);
	row_vector[N] x2 = dq[1,:].*x1;
	row_vector[N] x3 = cos(q[2,:]);
	row_vector[N] x4 = dq[1,:].*x3;
	row_vector[N] x5 = 9.81*x3;
	row_vector[N] x6 = a1*(ddq[2,:] + x2.*x4) + x5;
	row_vector[N] x7 = cos(q[3,:]);
	row_vector[N] x8 = 9.81*x1;
	row_vector[N] x9 = a1*(-((dq[2,:]).*(dq[2,:])) - ((x4).*(x4))) + x8;
	row_vector[N] x10 = x0.*x6 + x7.*x9;
	row_vector[N] x11 = ddq[2,:] + ddq[3,:];
	row_vector[N] x12 = dq[2,:].*x2;
	row_vector[N] x13 = ddq[1,:].*x3 - x12;
	row_vector[N] x14 = a1*(x12 - x13);
	row_vector[N] x15 = x0.*x4 + x2.*x7;
	row_vector[N] x16 = -x0;
	row_vector[N] x17 = x16.*x2 + x4.*x7;
	row_vector[N] x18 = dq[2,:] + dq[3,:];
	row_vector[N] x19 = params[25]*x15 + params[27]*x17 + params[28]*x18;
	row_vector[N] x20 = -x15;
	row_vector[N] x21 = ddq[1,:].*x1 + dq[2,:].*x4;
	row_vector[N] x22 = dq[3,:].*x17 + x0.*x13 + x21.*x7;
	row_vector[N] x23 = params[23]*x15 + params[24]*x17 + params[25]*x18;
	row_vector[N] x24 = dq[3,:].*x20 - x0.*x21 + x13.*x7;
	row_vector[N] x25 = params[24]*x22 + params[26]*x24 + params[27]*x11 - params[29]*x14 + params[31]*x10 + x18.*x23 + x19.*x20;
	row_vector[N] x26 = x16.*x9 + x6.*x7;
	row_vector[N] x27 = params[24]*x15 + params[26]*x17 + params[27]*x18;
	row_vector[N] x28 = params[23]*x22 + params[24]*x24 + params[25]*x11 + params[30]*x14 - params[31]*x26 + x17.*x19 - x18.*x27;
	row_vector[N] x29 = dq[2,:]*params[17] + params[14]*x2 + params[16]*x4;
	row_vector[N] x30 = dq[2,:]*params[16] + params[13]*x2 + params[15]*x4;
	row_vector[N] x31 = dq[2,:]*params[14] + params[12]*x2 + params[13]*x4;
	row_vector[N] x32 = -((x17).*(x17));
	row_vector[N] x33 = -((x15).*(x15));
	row_vector[N] x34 = x17.*x18;
	row_vector[N] x35 = x15.*x18;
	row_vector[N] x36 = params[25]*x22 + params[27]*x24 + params[28]*x11 + params[29]*x26 - params[30]*x10 + x15.*x27 - x17.*x23;
	row_vector[N] x37 = -((x18).*(x18));
	row_vector[N] x38 = x15.*x17;
//
	tau_hat[1,:] = ddq[1,:]*params[6] + dq[1,:]*params[11] + x1.*(ddq[2,:]*params[14] - dq[2,:].*x30 + params[12]*x21 + params[13]*x13 - params[20]*x5 + x16.*x25 + x28.*x7 + x29.*x4) + x3.*(-a1*(params[29]*(-x24 + x35) + params[30]*(x22 + x34) + params[31]*(x32 + x33) + params[32]*x14) + ddq[2,:]*params[16] + dq[2,:].*x31 + params[13]*x21 + params[15]*x13 + params[20]*x8 + x0.*x28 - x2.*x29 + x25.*x7);
	tau_hat[2,:] = a1*(x0.*(params[29]*(x32 + x37) + params[30]*(-x11 + x38) + params[31]*(x24 + x35) + params[32]*x10) + x7.*(params[29]*(x11 + x38) + params[30]*(x33 + x37) + params[31]*(-x22 + x34) + params[32]*x26)) + ddq[2,:]*params[17] + dq[2,:]*params[22] + params[14]*x21 + params[16]*x13 + params[18]*x5 - params[19]*x8 + x2.*x30 - x31.*x4 + x36;
	tau_hat[3,:] = dq[3,:]*params[33] + x36;
//
    r ~ cauchy(0, 1.0);
//	eig11 ~ cauchy(0, 1);
//	eig12 ~ cauchy(0, 1);
//	eig13 ~ cauchy(0, 1);
	r_1 ~ cauchy(0, 1.0);
	fv_1 ~ cauchy(0, 1.0);
	to_vector(I_1) ~ cauchy(0, 1);
	to_vector(I_2) ~ cauchy(0, 1);
	to_vector(I_3) ~ cauchy(0, 1);
	r_2 ~ cauchy(0, 1.0);
	fv_2 ~ cauchy(0, 1.0);
//	eig31 ~ cauchy(0, 1);
//	eig32 ~ cauchy(0, 1);
//	eig33 ~ cauchy(0, 1);
	r_3 ~ cauchy(0, 1.0);
	fv_3 ~ cauchy(0, 1.0);
	tau[1, :] ~ normal(tau_hat[1, :], r);
	tau[2, :] ~ normal(tau_hat[2, :], r);
	tau[3, :] ~ normal(tau_hat[3, :], r);
}

