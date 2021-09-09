data { 
	int<lower=0> N;
	int<lower=0> dof;
	matrix[3, N] q;
	matrix[3, N] dq;
	matrix[3, N] ddq;
	matrix[3, N] tau;
}
transformed data { 
	matrix[dof, N] sin_q = sin(q);
	matrix[dof, N] cos_q = cos(q);
	row_vector[N] x0 = sin_q[2,:];
	row_vector[N] x1 = dq[1,:].*x0;
	row_vector[N] x2 = sin_q[3,:];
	row_vector[N] x3 = -x2;
	row_vector[N] x4 = cos_q[3,:];
	row_vector[N] x5 = cos_q[2,:];
	row_vector[N] x6 = dq[1,:].*x5;
	row_vector[N] x7 = x1.*x3 + x4.*x6;
	row_vector[N] x8 = dq[2,:] + dq[3,:];
	row_vector[N] x9 = x7.*x8;
	row_vector[N] x10 = dq[2,:].*x1;
	row_vector[N] x11 = ddq[1,:].*x5 - x10;
	row_vector[N] x12 = ddq[1,:].*x0 + dq[2,:].*x6;
	row_vector[N] x13 = dq[3,:].*x7 + x11.*x2 + x12.*x4;
	row_vector[N] x14 = x1.*x4 + x2.*x6;
	row_vector[N] x15 = -x14;
	row_vector[N] x16 = dq[3,:].*x15 + x11.*x4 - x12.*x2;
	row_vector[N] x17 = x14.*x8;
	row_vector[N] x18 = -((x14).*(x14));
	row_vector[N] x19 = -((x7).*(x7));
	row_vector[N] x20 = 0.8*x10 - 0.8*x11;
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
		row_vector[N] x28  x2.*(0.8*ddq[2,:] + 0.8*x1.*x6 + (9.81*x5)) + (-0.8*((dq[2,:]).*(dq[2,:])) + (9.81*x0) - 0.8*((x6).*(x6))).*x4;
	row_vector[N] x29  params[24]*x13 + params[26]*x16 + params[27]*(ddq[2,:] + ddq[3,:]) - params[29]*x20 + params[31]*x28 + x15.*(params[25]*x14 + params[27]*x7 + params[28]*x8) + (params[23]*x14 + params[24]*x7 + params[25]*x8).*x8;
	row_vector[N] x30  (0.8*ddq[2,:] + 0.8*x1.*x6 + (9.81*x5)).*x4 + (-0.8*((dq[2,:]).*(dq[2,:])) + (9.81*x0) - 0.8*((x6).*(x6))).*x3;
	row_vector[N] x32  params[23]*x13 + params[24]*x16 + params[25]*(ddq[2,:] + ddq[3,:]) + params[30]*x20 - params[31]*x30 + (params[25]*x14 + params[27]*x7 + params[28]*x8).*x7 - (params[24]*x14 + params[26]*x7 + params[27]*x8).*x8;
	row_vector[N] x38  params[25]*x13 + params[27]*x16 + params[28]*(ddq[2,:] + ddq[3,:]) + params[29]*x30 - params[30]*x28 + x14.*(params[24]*x14 + params[26]*x7 + params[27]*x8) - (params[23]*x14 + params[24]*x7 + params[25]*x8).*x7;
	tau_hat[1,:]  ddq[1,:]*params[6] + dq[1,:]*params[11] + x0.*(ddq[2,:]*params[14] - dq[2,:].*(dq[2,:]*params[16] + params[13]*x1 + params[15]*x6) + params[12]*x12 + params[13]*x11 - params[20]*(9.81*x5) + x29.*x3 + x32.*x4 + (dq[2,:]*params[17] + params[14]*x1 + params[16]*x6).*x6) + x5.*(ddq[2,:]*params[16] + dq[2,:].*(dq[2,:]*params[14] + params[12]*x1 + params[13]*x6) + params[13]*x12 + params[15]*x11 + params[20]*(9.81*x0) - 0.8*params[29]*(-x16 + x17) - 0.8*params[30]*(x13 + x9) - 0.8*params[31]*(x18 + x19) - 0.8*params[32]*x20 - x1.*(dq[2,:]*params[17] + params[14]*x1 + params[16]*x6) + x2.*x32 + x29.*x4);
	tau_hat[2,:]  ddq[2,:]*params[17] + dq[2,:]*params[22] + params[14]*x12 + params[16]*x11 + params[18]*(9.81*x5) - params[19]*(9.81*x0) + x1.*(dq[2,:]*params[16] + params[13]*x1 + params[15]*x6) + 0.8*x2.*(params[29]*(x19 + (-((x8).*(x8)))) + params[30]*(-(ddq[2,:] + ddq[3,:]) + (x14.*x7)) + params[31]*(x16 + x17) + params[32]*x28) - (dq[2,:]*params[14] + params[12]*x1 + params[13]*x6).*x6 + x38 + 0.8*x4.*(params[29]*((ddq[2,:] + ddq[3,:]) + (x14.*x7)) + params[30]*(x18 + (-((x8).*(x8)))) + params[31]*(-x13 + x9) + params[32]*x30);
	tau_hat[3,:]  dq[3,:]*params[33] + x38;
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
