result = []

with open('w2v_pol_no_accents.txt', 'r') as f:
    text = f.readlines()

    tmp = ''
    flag = False

    for t in text:
        if flag:
            tmp = tmp.replace('\n', '')
            tmp += t
            result.append(tmp)
            flag = False
            tmp = ''
            continue 
        tmp = t
        if len(t) < 100:
            flag = True
        else:
            result.append(tmp)

with open('w2v_pol_no_accents_fix.txt', 'w') as f:
    for r in result:
        if len(r.split()) == 101:
            f.write('{}'.format(r))
