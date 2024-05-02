import matplotlib.pyplot as plt
import glob 

for fname in glob.glob('./*.log'): 
    file = open(fname, 'r')
    lines = file.readlines()
    data = []
    title = ""
    for line in lines:
        if "Size" in line and "GB/s" in line: data.append(line)

    labels = ["Kokkos", "Thrust", "Cuda", "Cuda (vl)"]
    nelmts = [float(line.split()[1]) for line in data]
    GBs    = [[float(GB) for GB in line.split()[3:]] for line in data]
    colors = ["royalblue", "goldenrod", "darkgreen", "darkgreen"]
    linestyles = ["-",  "-", "-", "--"]

    plt.figure()
    for i in range(0, len(GBs[0])):
        plt.semilogx(nelmts, [line[i] for line in GBs], linestyle=linestyles[i], color=colors[i], label=labels[i])
    plt.legend()
    plt.savefig(fname.split(".log")[0] + ".png")
