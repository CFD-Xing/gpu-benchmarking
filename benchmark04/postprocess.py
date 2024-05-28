import matplotlib.pyplot as plt
import glob 

for fname in glob.glob('./*.log'): 
    file = open(fname, 'r')
    lines = file.readlines()
    data = []
    title = ""
    for line in lines:
        if "nelmt" in line and "DOF/s" in line: data.append(line)
        if "NQ =" in line: title = line

    labels = ["Kokkos (Uncoales)", "Kokkos (Coales)", "Kokkos (QP-1D)", "Kokkos (QP-1D/Shared)", "cuBLAS", "Cuda (Uncoales)", "Cuda (Coales)", "Cuda (QP-MD)", "Cuda (QP-MD/Shared)", "Cuda (QP-1D)", "Cuda (QP-1D/Shared)"]
    nelmts = [float(line.split()[1]) for line in data]
    DOFs    = [[float(DOF) for DOF in line.split()[3:]] for line in data]
    colors = ["cornflowerblue", "cornflowerblue", "darkblue", "darkblue", "goldenrod", "greenyellow", "greenyellow", "darkgreen", "darkgreen", "darkgreen", "darkgreen"]
    linestyles = ["-", "--", "-.", ":", "-", "-", "--", "-", "--", "-.", ":"]

    plt.figure()
    for i in range(0, len(DOFs[0])):
        plt.semilogx(nelmts, [line[i] for line in DOFs], linestyle=linestyles[i], color=colors[i], label=labels[i])
    plt.legend()
    plt.ylim([0, 400])
    plt.xlabel('Number of elmt.')
    plt.ylabel('DOF (1e9/s)')
    plt.title(title)
    plt.savefig(fname.split(".log")[0] + ".png")
