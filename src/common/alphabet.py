def latin():
    def index_to_char(index):
        return str(chr(ord('A') + index))

    return list(map(index_to_char,range(0,26)))

def greek():
    return { 0: "π", 1: "α", 2: "β", 3: "σ", 4: "γ", 5: "δ", 6: "λ", 7: "ω", 8: "μ", 9: "ξ"}

def greek_no_unicode():
    return { 0: "pi", 1: "alpha", 2: "beta", 3: "sigma", 4: "gamma", 5: "delta", 6: "lambda", 7: "omega", 8: "mu", 9: "xi"}
