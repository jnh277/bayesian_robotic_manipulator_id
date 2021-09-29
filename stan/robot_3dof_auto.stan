data { 
	int<lower=0> N;
	int<lower=0> dof;
	matrix[dof, N] q;
	matrix[dof, N] dq;
	matrix[dof, N] ddq;
	matrix[dof, N] tau;
}
transformed data { 
	matrix[dof, N] sin_q = sin(q);
	matrix[dof, N] cos_q = cos(q);
	}
parameters { 
	real<lower=1e-6> r;
	real<lower=1e-6> m[dof];
	vector[3] r_com[dof];
	unit_vector[4] quat[dof];
	real<lower=0.0> fv[dof];
	real <lower=0.0> eig12;
	real <lower=0.0> eig13;
	real <lower=fmax(eig13-eig12,eig12-eig13), upper=eig12+eig13> eig11;
	real <lower=0.0> eig22;
	real <lower=0.0> eig23;
	real <lower=fmax(eig23-eig22,eig22-eig23), upper=eig22+eig23> eig21;
	real <lower=0.0> eig32;
	real <lower=0.0> eig33;
	real <lower=fmax(eig33-eig32,eig32-eig33), upper=eig32+eig33> eig31;
}
transformed parameters {
	vector[3] eigs[dof];
	matrix[3,3] R[dof];
	matrix[3,3] I[dof];
	vector[3] l[dof];
	real Lxx[dof];	real Lyy[dof];	real Lzz[dof];	real Lxy[dof];	real Lxz[dof];	real Lyz[dof];		eigs[1,1] = eig11; eigs[1,2] = eig12; eigs[1,3] = eig13;
	eigs[2,1] = eig21; eigs[2,2] = eig22; eigs[2,3] = eig23;
	eigs[3,1] = eig31; eigs[3,2] = eig32; eigs[3,3] = eig33;
	for (d in 1:dof){
		R[d,1,1] = quat[d,1]*quat[d,1] + quat[d,2]*quat[d,2] - quat[d,3]*quat[d,3] - quat[d,4]*quat[d,4]; R[d,1,2] = 2*(quat[d,2]*quat[d,3]-quat[d,1]*quat[d,4]); R[d,1,3] = 2*(quat[d,1]*quat[d,3]+quat[d,2]*quat[d,4]);
		R[d,2,1] = 2*(quat[d,2]*quat[d,3]+quat[d,1]*quat[d,4]); R[d,2,2] = quat[d,1]*quat[d,1] - quat[d,2]*quat[d,2] + quat[d,3]*quat[d,3] - quat[d,4]*quat[d,4]; R[d,2,3] = 2*(quat[d,3]*quat[d,4]-quat[d,1]*quat[d,2]);
		R[d,3,1] = 2*(quat[d,2]*quat[d,4]-quat[d,1]*quat[d,3]); R[d,3,2] = 2*(quat[d,1]*quat[d,2]+quat[d,3]*quat[d,4]); R[d,3,3] = quat[d,1]*quat[d,1] - quat[d,2]*quat[d,2] - quat[d,3]*quat[d,3] + quat[d,4]*quat[d,4];
		I[d] = diag_post_multiply(R[d],eigs[d]) * R[d]';
		l[d] = r_com[d] * m[d];
		Lxx[d] = I[d,1,1] + m[d]*pow(r_com[d,2],2) + m[d]*pow(r_com[d,3],2);
		Lyy[d] = I[d,2,2] + m[d]*pow(r_com[d,1],2) + m[d]*pow(r_com[d,3],2);
		Lzz[d] = I[d,3,3] + m[d]*pow(r_com[d,1],2) + m[d]*pow(r_com[d,2],2);
		Lxy[d] = I[d,1,2] - m[d]*r_com[d,1]*r_com[d,2];
		Lxz[d] = I[d,1,3] - m[d]*r_com[d,1]*r_com[d,3];
		Lyz[d] = I[d,2,3] - m[d]*r_com[d,2]*r_com[d,3];
	}
	row_vector[33] params = [Lxx[1], Lxy[1], Lxz[1], Lyy[1], Lyz[1], Lzz[1], l[1,1], l[1,2], l[1,3], m[1], fv[1], 
				Lxx[2], Lxy[2], Lxz[2], Lyy[2], Lyz[2], Lzz[2], l[2,1], l[2,2], l[2,3], m[2], fv[2], 
				Lxx[3], Lxy[3], Lxz[3], Lyy[3], Lyz[3], Lzz[3], l[3,1], l[3,2], l[3,3], m[3], fv[3]];
}
model {
	matrix[dof, N] tau_hat;
	row_vector[N] x0 = sin_q[2,:];
	row_vector[N] x1 = dq[1,:].*x0;
	row_vector[N] x2 = dq[2,:].*x1;
	row_vector[N] x3 = cos_q[2,:];
	row_vector[N] x4 = ddq[1,:].*x3 - x2;
	row_vector[N] x5 = 9.81*x0;
	row_vector[N] x6 = dq[1,:].*x3;
	row_vector[N] x7 = dq[2,:]*params[17] + params[14]*x1 + params[16]*x6;
	row_vector[N] x8 = ddq[1,:].*x0 + dq[2,:].*x6;
	row_vector[N] x9 = dq[2,:]*params[14] + params[12]*x1 + params[13]*x6;
	row_vector[N] x10 = 0.8*x2 - 0.8*x4;
	row_vector[N] x11 = ddq[2,:] + ddq[3,:];
	row_vector[N] x12 = cos_q[3,:];
	row_vector[N] x13 = sin_q[3,:];
	row_vector[N] x14 = x1.*x12 + x13.*x6;
	row_vector[N] x15 = -x13;
	row_vector[N] x16 = x1.*x15 + x12.*x6;
	row_vector[N] x17 = dq[2,:] + dq[3,:];
	row_vector[N] x18 = params[24]*x14 + params[26]*x16 + params[27]*x17;
	row_vector[N] x19 = -x14;
	row_vector[N] x20 = dq[3,:].*x19 + x12.*x4 - x13.*x8;
	row_vector[N] x21 = dq[3,:].*x16 + x12.*x8 + x13.*x4;
	row_vector[N] x22 = params[25]*x14 + params[27]*x16 + params[28]*x17;
	row_vector[N] x23 = 9.81*x3;
	row_vector[N] x24 = 0.8*ddq[2,:] + 0.8*x1.*x6 + x23;
	row_vector[N] x25 = -0.8*((dq[2,:]).*(dq[2,:])) + x5 - 0.8*((x6).*(x6));
	row_vector[N] x26 = x12.*x24 + x15.*x25;
	row_vector[N] x27 = params[23]*x21 + params[24]*x20 + params[25]*x11 + params[30]*x10 - params[31]*x26 + x16.*x22 - x17.*x18;
	row_vector[N] x28 = x14.*x17;
	row_vector[N] x29 = -((x16).*(x16));
	row_vector[N] x30 = -((x14).*(x14));
	row_vector[N] x31 = x16.*x17;
	row_vector[N] x32 = params[23]*x14 + params[24]*x16 + params[25]*x17;
	row_vector[N] x33 = x12.*x25 + x13.*x24;
	row_vector[N] x34 = params[24]*x21 + params[26]*x20 + params[27]*x11 - params[29]*x10 + params[31]*x33 + x17.*x32 + x19.*x22;
	row_vector[N] x35 = dq[2,:]*params[16] + params[13]*x1 + params[15]*x6;
	row_vector[N] x36 = x14.*x16;
	row_vector[N] x37 = -((x17).*(x17));
	row_vector[N] x38 = params[25]*x21 + params[27]*x20 + params[28]*x11 + params[29]*x26 - params[30]*x33 + x14.*x18 - x16.*x32;
//
	tau_hat[1,:] = ddq[1,:]*params[6] + dq[1,:]*params[11] + x0.*(ddq[2,:]*params[14] - dq[2,:].*x35 + params[12]*x8 + params[13]*x4 - params[20]*x23 + x12.*x27 + x15.*x34 + x6.*x7) + x3.*(ddq[2,:]*params[16] + dq[2,:].*x9 + params[13]*x8 + params[15]*x4 + params[20]*x5 - 0.8*params[29]*(-x20 + x28) - 0.8*params[30]*(x21 + x31) - 0.8*params[31]*(x29 + x30) - 0.8*params[32]*x10 - x1.*x7 + x12.*x34 + x13.*x27);
	tau_hat[2,:] = ddq[2,:]*params[17] + dq[2,:]*params[22] + params[14]*x8 + params[16]*x4 + params[18]*x23 - params[19]*x5 + x1.*x35 + 0.8*x12.*(params[29]*(x11 + x36) + params[30]*(x30 + x37) + params[31]*(-x21 + x31) + params[32]*x26) + 0.8*x13.*(params[29]*(x29 + x37) + params[30]*(-x11 + x36) + params[31]*(x20 + x28) + params[32]*x33) + x38 - x6.*x9;
	tau_hat[3,:] = dq[3,:]*params[33] + x38;
//
		r ~ cauchy(0, 1.0);
	for (d in 1:dof){
		r_com[d] ~ cauchy(0, 1.0);
		fv[d] ~ cauchy(0, 1.0);
	}
	eig11 ~ cauchy(0, 1);
	eig12 ~ cauchy(0, 1);
	eig13 ~ cauchy(0, 1);
	eig21 ~ cauchy(0, 1);
	eig22 ~ cauchy(0, 1);
	eig23 ~ cauchy(0, 1);
	eig31 ~ cauchy(0, 1);
	eig32 ~ cauchy(0, 1);
	eig33 ~ cauchy(0, 1);
	tau[1, :] ~ normal(tau_hat[1, :], r);
	tau[2, :] ~ normal(tau_hat[2, :], r);
	tau[3, :] ~ normal(tau_hat[3, :], r);
}
