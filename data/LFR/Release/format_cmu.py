raw_path = "size50_mu0.7_minc25_maxc25//community.dat"
format_path = "size50_mu0.7_minc25_maxc25//real_cmu.dat"

cmu_dict = {}
with open(raw_path, 'r') as raw_cmu:
    for line in raw_cmu.readlines():
        node, cmu = line.split()
        if int(cmu) in cmu_dict.keys():
            cmu_dict[int(cmu)].append(node)
        else:
            cmu_dict[int(cmu)] = [node]

with open(format_path, 'w') as f:
    for cmu_id, nodes in cmu_dict.items():
        f.write(' '.join(nodes))
        f.write("\n")