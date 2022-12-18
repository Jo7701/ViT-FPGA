import numpy as np

TILE_SIZE = 4

def load_regs1(a, b, a_regs, b_regs, i, j, k):
    for outer in range(TILE_SIZE):
        for inner in range(TILE_SIZE):
            a_hidx = outer + i*TILE_SIZE
            a_widx = inner + k*TILE_SIZE
            b_hidx = inner + k*TILE_SIZE
            b_widx = outer + j*TILE_SIZE

            a_regs[outer, inner] = a[a_hidx, a_widx]
            b_regs[inner, outer] = b[b_hidx, b_widx]

def load_regs2(c, layer1_temp, c_regs, d_regs, i, k):
    for outer in range(TILE_SIZE):
        for inner in range(TILE_SIZE):
            c_hidx = outer + i*TILE_SIZE
            c_widx = inner + k*TILE_SIZE
            d_hidx = inner + k*TILE_SIZE
            d_widx = outer

            c_regs[outer, inner] = c[c_hidx, c_widx]
            d_regs[inner, outer] = layer1_temp[d_hidx, d_widx]

def compute(a_regs, b_regs, out_regs, set_not_increment):
    for tk in range(TILE_SIZE):
        for ti in range(TILE_SIZE):
            for tj in range(TILE_SIZE):
                mul = a_regs[ti, tk] * b_regs[tk, tj]
                if set_not_increment and tk==0:
                    out_regs[ti, tj] = mul
                else:
                    out_regs[ti, tj] += mul

def store1(out_regs, layer1_temp, i):
    for ti in range(TILE_SIZE):
        for tj in range(TILE_SIZE):
            hidx = i*TILE_SIZE+ti
            widx = tj
            layer1_temp[hidx, widx] = out_regs[ti, tj]

def store2(out_regs, output, i, j):
    for ti in range(TILE_SIZE):
        for tj in range(TILE_SIZE):
            hidx = i*TILE_SIZE+ti
            widx = j*TILE_SIZE+tj
            output[hidx, widx] = out_regs[ti, tj]

def main():
    a = np.random.randint(0, 100, (64,64))
    b = np.random.randint(0, 100, (64, 64))
    c = np.random.randint(0, 100, (64, 64))
    output = np.zeros((64, 64))

    a_regs = np.zeros((TILE_SIZE, TILE_SIZE))
    b_regs = np.zeros((TILE_SIZE, TILE_SIZE))
    out1_regs = np.zeros((TILE_SIZE,TILE_SIZE))
    layer1_temp = np.zeros((64, TILE_SIZE))

    c_regs = np.zeros((TILE_SIZE, TILE_SIZE))
    d_regs = np.zeros((TILE_SIZE, TILE_SIZE))
    out2_regs = np.zeros((TILE_SIZE,TILE_SIZE))

    for j in range(64//TILE_SIZE):
        for i in range(64//TILE_SIZE):
            for k in range(64//TILE_SIZE):
                load_regs1(a, b, a_regs, b_regs, i, j, k)
                compute(a_regs, b_regs, out1_regs, k==0)
                store1(out1_regs, layer1_temp, i)

        for l in range(64//TILE_SIZE):
            for m in range(64//TILE_SIZE):
                load_regs2(c, layer1_temp, c_regs, d_regs, l, m)
                compute(c_regs, d_regs, out2_regs, m==0)
                store2(out2_regs, output, l, j)

    golden = np.matmul(c, np.matmul(a,b))
    print(output==golden)


if __name__ == "__main__":
    main()