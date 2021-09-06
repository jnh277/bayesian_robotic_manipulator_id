import re

def c_to_stan(code_str, dof, num_params):
    code_str = code_str.replace("  ", "\t")     # replace double spaces with tabs
    for d in range(dof-1,-1,-1):
        code_str = code_str.replace("q["+str(d)+"]", "q["+str(d+1)+",:]")
        code_str = code_str.replace("dq[" + str(d) + "]", "ddq[" + str(d + 1) + ",:]")
        code_str = code_str.replace("dq[" + str(d) + "]", "ddq[" + str(d + 1) + ",:]")
        code_str = code_str.replace("tau_out[" + str(d) + "]", "tau_hat[" + str(d + 1) + ",:]")

    idx = code_str.index("double x0")
    code_str = code_str[idx:]

    # remove the stupid variables
    sub_strs = re.findall('\tdouble x\d* = -parms\[\d*\];\n', code_str)
    for sub_str in sub_strs:
        x_remove = re.findall('x\d*', sub_str)
        param_replace = re.findall('-parms\[\d*\]', sub_str)
        code_str = code_str.replace(sub_str, "")
        code_str = code_str.replace(x_remove[0]+" ", param_replace[0]+" ")
        code_str = code_str.replace(x_remove[0] + "*", param_replace[0]+"*")
        code_str = code_str.replace("*"+x_remove[0]+" ", "*"+param_replace[0]+" ")
        code_str = code_str.replace("*" + x_remove[0] + ";", "*" + param_replace[0] + ";")

    code_str = code_str.replace("double", "row_vector[N]")

    for i in range(num_params-1,-1,-1):
        code_str = code_str.replace("parms["+str(i)+"]", "params["+str(i+1)+"]")

    sub_strs = re.findall('x\d*\*x\d*|x\d*\*\(|x\d*\)\*',code_str)
    for sub_str in sub_strs:
        new_str = sub_str.replace('*','.*')
        code_str = code_str.replace(sub_str, new_str)

    sub_strs = re.findall('x\d*\*q\[\d*,:\]|q\[\d*,:\]\*x\d|q\[\d*,:\]\*\(|q\[\d*,:\]\*q\[\d*,:\]|q\[\d*,:\]\*\)',code_str)
    for sub_str in sub_strs:
        new_str = sub_str.replace('*','.*')
        code_str = code_str.replace(sub_str, new_str)

    sub_strs = re.findall('x\d*\*d*q\[\d*,:\]|d*q\[\d*,:\]\*x\d|d*q\[\d*,:\]\*\(|d*q\[\d*,:\]\)\*|d*q\[\d*,:\]\*q\[\d*,:\]|d*q\[\d*,:\]\*d*q\[\d*,:\]|q\[\d*,:\]\*d*q\[\d*,:\]',code_str)
    for sub_str in sub_strs:
        new_str = sub_str.replace('*','.*')
        code_str = code_str.replace(sub_str, new_str)

    # for d in range(dof-1,-1,-1):        # get ride of sign(dq[blah,:]) and input this as data instead
    #     code_str = code_str.replace("sign(dq["+str(d+1)+",:])", "sign_dq["+str(d+1)+",:]")
    for d in range(dof-1,-1,-1):        # get ride of sign(dq[blah,:]) and input this as data instead
        code_str = code_str.replace("sign(dq[" + str(d + 1) + ",:])", "sign_dq[" + str(d + 1) + ",:]")
        code_str = code_str.replace("cos(q["+str(d+1)+",:])", "cos_q["+str(d+1)+",:]")
        code_str = code_str.replace("sin(q[" + str(d + 1) + ",:])", "sin_q[" + str(d + 1) + ",:]")

    idx = code_str.index("return;")
    code_str = code_str[:idx]

    ## split based on where param first gets called
    for line in code_str.split('\n'):
        if line.find('param') >= 0:
            first_param_line = line
            break
    idx = code_str.index(first_param_line)

    return code_str[:idx], code_str[idx:]


def create_param_block(dof, frictionmodel=None,driveinertiamodel=None):
    code_str = "parameters { \n"
    code_str += "\treal<lower=1e-6> r;\n"
    code_str += "\treal<lower=1e-6> m[dof];\n"
    code_str += "\tvector[3] r_com[dof];\n"
    code_str += "\tunit_vector[4] quat[dof];\n"
    if driveinertiamodel is not None:
        if driveinertiamodel.lower() == 'simplified':
            code_str += "\treal<lower=0.0> Ia[dof];\n"
    if frictionmodel is not None:
        if 'viscous' in {s.lower() for s in frictionmodel}:
            code_str += "\treal<lower=0.0> fv[dof];\n"
        if 'coulomb' in {s.lower() for s in frictionmodel}:
            code_str += "\treal<lower=0.0> fc[dof];\n"
        if 'offset' in {s.lower() for s in frictionmodel}:
            code_str += "\treal<lower=0.0> fo[dof];\n"

    for d in range(1,dof+1):
        code_str += "\treal <lower=0.0> eig"+str(d)+"2;\n"
        code_str += "\treal <lower=0.0> eig"+str(d)+"3;\n"
        code_str += "\treal <lower=fmax(eig"+str(d)+"3-eig"+str(d)+"2,eig"+str(d)+"2-eig"+str(d)+"3), upper=eig"+str(d)+"2+eig"+str(d)+"3> eig" + str(d) + "1;\n"

    code_str += "}\n"
    return code_str

def create_trans_params_block(dof, num_params, frictionmodel=None,driveinertiamodel=None):
    code_str = "transformed parameters {\n"
    code_str += "\tvector[3] eigs[dof];\n"
    code_str += "\tmatrix[3,3] R[dof];\n"
    code_str += "\tmatrix[3,3] I[dof];\n"
    code_str += "\tvector[3] l[dof];\n"
    code_str += "\treal Lxx[dof];\t"
    code_str += "real Lyy[dof];\t"
    code_str += "real Lzz[dof];\t"
    code_str += "real Lxy[dof];\t"
    code_str += "real Lxz[dof];\t"
    code_str += "real Lyz[dof];\t"

    for d in range(1,dof+1):
        code_str += "\teigs["+str(d)+",1] = eig"+str(d)+"1; eigs["+str(d)+",2] = eig"+str(d)+"2; eigs["+str(d)+",3] = eig"+str(d)+"3;\n"

    code_str += "\tfor (d in 1:dof){\n"
    code_str += "\t\tR[d,1,1] = quat[d,1]*quat[d,1] + quat[d,2]*quat[d,2] - quat[d,3]*quat[d,3] - quat[d,4]*quat[d,4]; R[d,1,2] = 2*(quat[d,2]*quat[d,3]-quat[d,1]*quat[d,4]); R[d,1,3] = 2*(quat[d,1]*quat[d,3]+quat[d,2]*quat[d,4]);\n"
    code_str += "\t\tR[d,2,1] = 2*(quat[d,2]*quat[d,3]+quat[d,1]*quat[d,4]); R[d,2,2] = quat[d,1]*quat[d,1] - quat[d,2]*quat[d,2] + quat[d,3]*quat[d,3] - quat[d,4]*quat[d,4]; R[d,2,3] = 2*(quat[d,3]*quat[d,4]-quat[d,1]*quat[d,2]);\n"
    code_str += "\t\tR[d,3,1] = 2*(quat[d,2]*quat[d,4]-quat[d,1]*quat[d,3]); R[d,3,2] = 2*(quat[d,1]*quat[d,2]+quat[d,3]*quat[d,4]); R[d,3,3] = quat[d,1]*quat[d,1] - quat[d,2]*quat[d,2] - quat[d,3]*quat[d,3] + quat[d,4]*quat[d,4];\n"
    code_str += "\t\tI[d] = diag_post_multiply(R[d],eigs[d]) * R[d]';\n"
    code_str += "\t\tl[d] = r_com[d] * m[d];\n"
    code_str += "\t\tLxx[d] = I[d,1,1] + m[d]*pow(r_com[d,2],2) + m[d]*pow(r_com[d,3],2);\n"
    code_str += "\t\tLyy[d] = I[d,2,2] + m[d]*pow(r_com[d,1],2) + m[d]*pow(r_com[d,3],2);\n"
    code_str += "\t\tLzz[d] = I[d,3,3] + m[d]*pow(r_com[d,1],2) + m[d]*pow(r_com[d,2],2);\n"
    code_str += "\t\tLxy[d] = I[d,1,2] - m[d]*r_com[d,1]*r_com[d,2];\n"
    code_str += "\t\tLxz[d] = I[d,1,3] - m[d]*r_com[d,1]*r_com[d,3];\n"
    code_str += "\t\tLyz[d] = I[d,2,3] - m[d]*r_com[d,2]*r_com[d,3];\n"
    code_str += "\t}\n"

    code_str += "\trow_vector["+str(num_params)+"] params = ["
    for d in range(1,dof+1):
        code_str += "Lxx["+str(d)+"], Lxy["+str(d)+"], Lxz["+str(d)+"], Lyy["+str(d)+"], Lyz["+str(d)+"], Lzz["+str(d)+"], l["+str(d)+",1], l["+str(d)+",2], l["+str(d)+",3], m["+str(d)+"]"
        if driveinertiamodel is not None:
            if driveinertiamodel.lower() == 'simplified':
                code_str += ", Ia["+str(d)+"]"
        if frictionmodel is not None:
            if 'viscous' in {s.lower() for s in frictionmodel}:
                code_str += ", fv["+str(d)+"]"
            if 'coulomb' in {s.lower() for s in frictionmodel}:
                code_str += ", fc["+str(d)+"]"
            if 'offset' in {s.lower() for s in frictionmodel}:
                code_str += ", fo["+str(d)+"]"
        if d != dof:
            code_str += ", \n\t\t\t\t"
    code_str += '];\n'

    code_str +="}\n"
    return code_str

def create_model_block(dof, c_code_str,frictionmodel=None,driveinertiamodel=None):
    code_str = "model {\n"
    code_str += "\tmatrix[dof, N] tau_hat;\n"
    code_str +="\t"+c_code_str
    code_str += "\tr ~ cauchy(0, 1.0);\n"
    code_str += "\tfor (d in 1:dof){\n"
    code_str += "\t\tr_com[d] ~ cauchy(0, 1.0);\n"
    if driveinertiamodel is not None:
        if driveinertiamodel.lower() == 'simplified':
            code_str += "\t\tIa[d] ~ cauchy(0, 1.0);\n"
    if frictionmodel is not None:
        if 'viscous' in {s.lower() for s in frictionmodel}:
            code_str += "\t\tfv[d] ~ cauchy(0, 1.0);\n"
        if 'coulomb' in {s.lower() for s in frictionmodel}:
            code_str += "\t\tfc[d] ~ cauchy(0, 1.0);\n"
        if 'offset' in {s.lower() for s in frictionmodel}:
            code_str += "\t\tfo[d] ~ cauchy(0, 1.0);\n"
    code_str += "\t}\n"
    for d in range(1,dof+1):
        code_str += "\teig"+str(d)+"1 ~ cauchy(0, 1);\n"
        code_str += "\teig"+str(d)+"2 ~ cauchy(0, 1);\n"
        code_str += "\teig"+str(d)+"3 ~ cauchy(0, 1);\n"

    for d in range(1, dof+1):
        code_str += "\ttau["+str(d)+", :] ~ normal(tau_hat["+str(d)+", :], r);\n"
    code_str += "}\n"

    return code_str

def create_data_block():
    code_str = "data { \n"
    code_str += "\tint<lower=0> N;\n"
    code_str += "\tint<lower=0> dof;\n"
    code_str += "\tmatrix[3, N] q;\n"
    code_str += "\tmatrix[3, N] dq;\n"
    code_str += "\tmatrix[3, N] ddq;\n"
    code_str += "\tmatrix[3, N] tau;\n"
    code_str += "}\n"
    return code_str

def create_transformed_data_block(extra_trans_code):
    code_str = "transformed data { \n"
    code_str += "\tmatrix[dof, N] sin_q = sin(q);\n"
    code_str += "\tmatrix[dof, N] cos_q = cos(q);\n"
    code_str += "\t"+extra_trans_code
    code_str += "}\n"
    return code_str

def create_stan_model(dof, num_params, tau_code_str, filename, frictionmodel=None,driveinertiamodel=None):
    code_str = create_data_block()
    trans_data_part, model_part = c_to_stan(tau_code_str, dof, num_params)
    code_str += create_transformed_data_block(trans_data_part)
    code_str += create_param_block(dof, frictionmodel, driveinertiamodel)
    code_str += create_trans_params_block(dof, num_params, frictionmodel, driveinertiamodel)
    code_str += create_model_block(dof, model_part, frictionmodel, driveinertiamodel)

    with open(filename, 'w') as f:
        f.write(code_str)


dof = 3
print(create_trans_params_block(dof, 33,frictionmodel={'viscous'},driveinertiamodel='simplified'))
print(create_param_block(dof, frictionmodel={'viscous'},driveinertiamodel='simplified'))
# print(create_param_block(dof, frictionmodel={'viscous','Coulomb','offset'},driveinertiamodel='simplified'))

