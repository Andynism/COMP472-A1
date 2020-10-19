def latin():
    def index_to_char(index):
        return str(chr(ord('A') + index))

    return list(map(index_to_char,range(0,26)))

def greek():
    return ["π", "α", "β", "σ", "γ", "δ", "λ", "ω", "μ", "ξ"]

def greek_no_unicode():
    return ["pi", "alpha", "beta", "sigma", "gamma", "delta", "lambda", "omega", "mu", "xi"]
