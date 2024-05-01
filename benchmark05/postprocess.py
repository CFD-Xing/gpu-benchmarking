import matplotlib.pyplot as plt
import glob 

for fname in glob.glob('./*.log'): 
    file = open(fname, 'r')
    lines = file.readlines()
    data = []
    title = ""
    for line in lines:
        if "nelmt" in line and "GB/s" in line: data.append(line)
        if "NQ =" in line: title = line

    labels = ["Kokkos (Uncoales)", "Kokkos (Coales)", "Kokkos (QP)", "Kokkos (QP/Shared)", "cuBLAS", "Cuda (Uncoales)", "Cuda (Coales)", "Cuda (QP)", "Cuda (QP/Shared)", "Cuda (QP-1D)", "Cuda (QP-1D/Shared)"]
    nelmts = [float(line.split()[1]) for line in data]
    GBs    = [[float(GB) for GB in line.split()[3:]] for line in data]
    colors = ["royalblue", "royalblue", "mediumblue", "mediumblue", "goldenrod", "yellowgreen", "yellowgreen", "darkgreen", "darkgreen", "darkgreen", "darkgreen"]
    linestyles = ["-", "--", "-", "--", "-", "-", "--", "-", "--", "-.", ":"]

    plt.figure()
    for i in range(0, len(GBs[0])):
        plt.semilogx(nelmts, [line[i] for line in GBs], linestyle=linestyles[i], color=colors[i], label=labels[i])
    plt.legend()
    plt.title(title)
    plt.savefig(fname.split(".log")[0] + ".png")
