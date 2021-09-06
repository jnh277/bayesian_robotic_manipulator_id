data {
    int<lower=0> N;
    int<lower=0> dof;
    matrix[3, N] q;
    matrix[3, N] dq;
    matrix[3, N] ddq;
    matrix[3, N] tau;
    real a1;
    real d0;
}

parameters {
	real<lower=1e-6> r;         // measurement standard deviations
	real<lower=1e-6> m[dof];    // mass of each link
	vector[3] r_com[dof];       // center of mass of each link
	real<lower=0.0> fv[dof];    // viscous friction term for each link
	unit_vector[4] quat[dof];   // quaternions describing inertia tensor rotation for each link
	// now separate eigen values for inertia tensors so that constraints can be imposed
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

	vector[3] eigs[dof];    // vectors of eigen values put together
    matrix[3,3] R[dof];     // rotation matrix for each link describing rotation of inertia
	matrix[3,3] I[dof];     // inertia about com for each link
	vector[3] l[dof];            // mass times position of com for each link
    real Lxx[dof];
    real Lyy[dof];
    real Lzz[dof];
    real Lxy[dof];
    real Lxz[dof];
    real Lyz[dof];

	eigs[1,1] = eig11; eigs[1,2] = eig12; eigs[1,3] = eig13;
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
	eig11 ~ cauchy(0, 1);
	eig12 ~ cauchy(0, 1);
	eig13 ~ cauchy(0, 1);
    eig21 ~ cauchy(0, 1);
	eig22 ~ cauchy(0, 1);
	eig23 ~ cauchy(0, 1);
    eig31 ~ cauchy(0, 1);
	eig32 ~ cauchy(0, 1);
	eig33 ~ cauchy(0, 1);
	for (d in 1:dof){
	    r_com[d] ~ cauchy(0, 1.0);
	    fv[d] ~ cauchy(0, 1.0);
	}
	tau[1, :] ~ normal(tau_hat[1, :], r);
	tau[2, :] ~ normal(tau_hat[2, :], r);
	tau[3, :] ~ normal(tau_hat[3, :], r);
}

