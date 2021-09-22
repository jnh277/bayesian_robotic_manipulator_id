import re

def compress_code(code_str):
    tmp = re.split('\n',code_str)
    LHS = []
    RHS = []
    var_list = []
    count=0
    for i, line in enumerate(tmp):
        if '=' in line:
            left, right = re.split('=', line)
            LHS.append(left)
            RHS.append(right)
            z = re.search('x\d*',LHS[count])
            count += 1
            if z:
                var_list.append(z.group())

    LHS_new = []
    RHS_new = []
    var_list_new = []
    for i, var in enumerate(var_list):
        if len(RHS[i]) < 100:
            for j in range(i+1,len(RHS)):
                RHS[j] = re.sub(var+'(?!\d)', '('+RHS[i][1:-1]+')',RHS[j])
        else:
            var_list_new.append(var)
            LHS_new.append(LHS[i])
            RHS_new.append(RHS[i])

    for k in range(i+1, len(LHS)):
        LHS_new.append(LHS[k])
        RHS_new.append(RHS[k])

    code_str_new = ""
    for i in range(len(LHS_new)):
        code_str_new += LHS_new[i] + '=' + RHS_new[i] +"\n"

    return code_str_new


def compress_code2(LHS_np, RHS_np, LHS_p, RHS_p):
    # compress the _np lines
    LHS_np_new = []
    RHS_np_new = []
    for i in range(len(LHS_np)-1):
        z = re.search('x\d*',LHS_np[i])        # look for a variable name
        if z: # if variable see if we want to compress
            var = z.group()
            if len(RHS_np[i]) < 100:    # if RHS is small then compress
                for j in range(i+1,len(RHS_np)):
                    RHS_np[j] = re.sub(var+'(?!\d)', '('+RHS_np[i][1:-1]+')',RHS_np[j])
                # also need to make substitutiones in the _p lines
                for j in range(len(RHS_p)):
                    RHS_p[j] = re.sub(var+'(?!\d)', '('+RHS_np[i][1:-1]+')',RHS_p[j])

            else:   # else make this a new line in code
                # var_list_new.append(var)
                LHS_np_new.append(LHS_np[i])
                RHS_np_new.append(RHS_np[i])
        else:   # if doesnt contain var then copy into LHS new and RHS new
            LHS_np_new.append(LHS_np[i])
            RHS_np_new.append(RHS_np[i])
    # copy across final line
    for k in range(i+1, len(LHS_np)):
        LHS_np_new.append(LHS_np[k])
        RHS_np_new.append(RHS_np[k])

    # compress the _p lines
    LHS_p_new = []
    RHS_p_new = []
    for i in range(len(LHS_p)-1):
        z = re.search('x\d*',LHS_p[i])        # look for a variable name
        if z: # if variable see if we want to compress
            var = z.group()
            if len(RHS_p[i]) < 100:    # if RHS is small then compress
                for j in range(i+1,len(RHS_p)):
                    RHS_p[j] = re.sub(var+'(?!\d)', '('+RHS_p[i][1:-1]+')',RHS_p[j])

            else:   # else make this a new line in code
                # var_list_new.append(var)
                LHS_p_new.append(LHS_p[i])
                RHS_p_new.append(RHS_p[i])
        else:   # if doesnt contain var then copy into LHS new and RHS new
            LHS_p_new.append(LHS_p[i])
            RHS_p_new.append(RHS_p[i])
    # copy across final line
    for k in range(i+1, len(LHS_p)):
        LHS_p_new.append(LHS_p[k])
        RHS_p_new.append(RHS_p[k])

    # put everything back together again
    code_p = ""
    code_np = ""
    for i in range(len(LHS_np_new)):
        code_np += LHS_np_new[i] + '=' + RHS_np_new[i] +"\n"

    for i in range(len(LHS_p_new)):
        code_p += LHS_p_new[i] + '=' + RHS_p_new[i] +"\n"
    return code_np, code_p

def sort_code_str(code_str):
    tmp = re.split('\n',code_str)
    LHS = []
    RHS = []
    var_list = []
    count = 0
    for i, line in enumerate(tmp):
        if '=' in line:
            left, right = re.split('=', line)
            LHS.append(left)
            RHS.append(right)
            z = re.search('x\d*',LHS[count])
            count +=1
            if z:
                var_list.append(z.group())

    param_deriv_list = ['param']
    LHS_np = []
    LHS_p = []
    RHS_np = []
    RHS_p = []

    for i in range(len(var_list)):
        match = False
        for p in param_deriv_list:
            if re.search(p+'(?!\d)', RHS[i]):
                match = True
                break
        if match:
            param_deriv_list.append(var_list[i])
            LHS_p.append(LHS[i])
            RHS_p.append(RHS[i])
        else:
            LHS_np.append(LHS[i])
            RHS_np.append(RHS[i])

    for k in range(i+1,len(LHS)):
        LHS_p.append(LHS[k])
        RHS_p.append(RHS[k])

    code_p = ""
    code_np = ""
    for i in range(len(LHS_np)):
        code_np += LHS_np[i] + '=' + RHS_np[i] +"\n"

    for i in range(len(LHS_p)):
        code_p += LHS_p[i] + '=' + RHS_p[i] +"\n"

    return code_np, code_p, LHS_np, RHS_np, LHS_p, RHS_p

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
        code_str = re.sub(x_remove[0]+'(?!\d)',param_replace[0],code_str)

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

    for d in range(dof-1,-1,-1):        # get ride of sign(dq[blah,:]) and input this as data instead
        code_str = code_str.replace("sign(dq[" + str(d + 1) + ",:])", "sign_dq[" + str(d + 1) + ",:]")
        code_str = code_str.replace("cos(q["+str(d+1)+",:])", "cos_q["+str(d+1)+",:]")
        code_str = code_str.replace("sin(q[" + str(d + 1) + ",:])", "sin_q[" + str(d + 1) + ",:]")

    idx = code_str.index("return;")
    code_str = code_str[:idx]


    # attempt to split more intelligently
    _, _, LHS_np, RHS_np, LHS_p, RHS_p = sort_code_str(code_str)
    trans_data_code, model_code = compress_code2(LHS_np, RHS_np, LHS_p, RHS_p)


    # model_code_new = compress_code(model_code)
    # trans_data_code_new = compress_code(trans_data_code)  # can't compress this as it doesnt feed through to model code

    return trans_data_code, model_code


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
    code_str += "\tmatrix[dof, N] q;\n"
    code_str += "\tmatrix[dof, N] dq;\n"
    code_str += "\tmatrix[dof, N] ddq;\n"
    code_str += "\tmatrix[dof, N] tau;\n"
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

