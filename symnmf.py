
import sys
import numpy as np
from symnmfmodule import sym, ddg, norm, symnmf

def main():
    np.random.seed(1234)
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        return

    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_name = sys.argv[3]

    try:
        data = np.loadtxt(file_name, delimiter=',')
    except:
        print("An Error Has Occurred")
        return

    if goal == "sym":
        result = sym(data)
    elif goal == "ddg":
        result = ddg(data)
    elif goal == "norm":
        result = norm(data)
    elif goal == "symnmf":
        similarity_matrix = sym(data)
        initial_H = np.random.uniform(0, 2 * np.sqrt(np.mean(similarity_matrix) / k), (similarity_matrix.shape[0], k))
        result = symnmf(similarity_matrix, initial_H, k)
    else:
        print("An Error Has Occurred")
        return

    np.savetxt(sys.stdout, result, delimiter=',', fmt='%.4f')

if __name__ == "__main__":
    main()
