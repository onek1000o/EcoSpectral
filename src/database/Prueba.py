D, V, I = 5000, 5000, 5000

# Procesamiento por lotes (batch processing)
for  i in range(20000):

    if D == 0 and V == 0 and I == 0:
        print(f" -------------------------> D : {D}, V : {V} , I : {I}")
        break
    elif i % 2 == 0 and D > 0:
        D = D-1
    elif i % 3 != 0 and V > 0:
        V = V-1
    elif I >= 0:
        I = I-1

    print(i)