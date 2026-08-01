class CSF:
    def __init__(self, nterms, terms):
        """
        nterms: int — number of terms in the CSF
        terms: list of tuples — each term is (coefficient: float, spin_a: list[int], spin_b: list[int])
        """
        self.nterms = nterms
        self.terms = terms

    def __repr__(self):
        return f"CSF(nterms={self.nterms}, terms={self.terms})"


def ne0spin1(paired, unpaired):
    if len(unpaired) != 0:
        raise ValueError("error in ne0spin1, incorrect unpaired length")

    terms = [
        (1.0, paired.copy(), paired.copy())
    ]
    return CSF(1, terms)


def ne1spin2(paired, unpaired):
    if len(unpaired) != 1:
        raise ValueError("error in ne1spin2, incorrect unpaired length")

    terms = [
        (1.0, paired.copy() + unpaired, paired.copy())
    ]
    return CSF(1, terms)


def ne2spin1(paired, unpaired):
    if len(unpaired) != 2:
        raise ValueError("error in ne2spin1, incorrect unpaired length")

    terms = [
        (0.70710678118654746, paired.copy() + [unpaired[0]], paired.copy() + [unpaired[1]]),
        (0.70710678118654746, paired.copy() + [unpaired[1]], paired.copy() + [unpaired[0]])
    ]
    return CSF(2, terms)


def ne2spin3(paired, unpaired):
    if len(unpaired) != 2:
        raise ValueError("error in ne2spin3, incorrect unpaired length")

    terms = [
        (1.0, paired.copy() + unpaired, paired.copy())
    ]
    return CSF(1, terms)


def ne3spin2_from_singlet(paired, unpaired):
    if len(unpaired) != 3:
        raise ValueError("error in ne3spin2, incorrect unpaired length")

    terms = [
        (0.70710678118654746, paired.copy() + [unpaired[0], unpaired[2]], paired.copy() + [unpaired[1]]),
        #(-0.70710678118654746, paired.copy() + [unpaired[0], unpaired[2]], paired.copy() + [unpaired[1]]),
        (0.70710678118654746, paired.copy() + [unpaired[1], unpaired[2]], paired.copy() + [unpaired[0]])
    ]
    return CSF(2, terms)

def ne3spin2_from_triplet(paired, unpaired):
    if len(unpaired) != 3:
        raise ValueError("error in ne3spin2, incorrect unpaired length")

    terms = [
        (-0.40824829046386307, paired.copy() + [unpaired[1], unpaired[2]], paired.copy() + [unpaired[0]]),
        (0.40824829046386307, paired.copy() + [unpaired[0], unpaired[2]], paired.copy() + [unpaired[1]]),
        (0.81649658092772615, paired.copy() + [unpaired[0], unpaired[1]], paired.copy() + [unpaired[2]])
    ]
    return CSF(3, terms)

def ne3spin4(paired, unpaired):
    if len(unpaired) != 3:
        raise ValueError("error in ne3spin4, incorrect unpaired length")

    terms = [
        (1.0, paired.copy() + unpaired, paired.copy())
    ]
    return CSF(1, terms)


def ne4spin1_a(paired, unpaired):
    if len(unpaired) != 4:
        raise ValueError("error in ne4spin1, incorrect unpaired length")

    terms = [
        (0.5, paired.copy() + [unpaired[0], unpaired[2]], paired.copy() + [unpaired[1], unpaired[3]]),
        (0.5, paired.copy() + [unpaired[0], unpaired[3]], paired.copy() + [unpaired[1], unpaired[2]]),
        (0.5, paired.copy() + [unpaired[1], unpaired[2]], paired.copy() + [unpaired[0], unpaired[3]]),
        (0.5, paired.copy() + [unpaired[1], unpaired[3]], paired.copy() + [unpaired[0], unpaired[2]])
    ]
    return CSF(4, terms)

def ne4spin1_b(paired, unpaired):
    if len(unpaired) != 4:
        raise ValueError("error in ne4spin1, incorrect unpaired length")
    terms = [
        (0.57735026918962584, paired.copy() + [unpaired[0], unpaired[1]], paired.copy() + [unpaired[2], unpaired[3]]),
        (0.57735026918962584, paired.copy() + [unpaired[2], unpaired[3]], paired.copy() + [unpaired[0], unpaired[1]]),
        (0.28867513459481292, paired.copy() + [unpaired[0], unpaired[2]], paired.copy() + [unpaired[1], unpaired[3]]),
        (0.28867513459481292, paired.copy() + [unpaired[1], unpaired[3]], paired.copy() + [unpaired[0], unpaired[2]]),
        (-0.28867513459481292, paired.copy() + [unpaired[0], unpaired[3]], paired.copy() + [unpaired[1], unpaired[2]]),
        (-0.28867513459481292, paired.copy() + [unpaired[1], unpaired[2]], paired.copy() + [unpaired[0], unpaired[3]])
    ]
    return CSF(6, terms)

def ne4spin3_a(paired, unpaired):
    if len(unpaired) != 4:
        raise ValueError("error in ne4spin3, incorrect unpaired length")

    terms = [
        (0.70710678118654746, paired.copy() + [unpaired[0], unpaired[2], unpaired[3]], paired.copy() + [unpaired[1]]),
        (0.70710678118654746, paired.copy() + [unpaired[1], unpaired[2], unpaired[3]], paired.copy() + [unpaired[0]])
    ]
    return CSF(2, terms)

def ne4spin3_b(paired, unpaired):
    if len(unpaired) != 4:
        raise ValueError("error in ne4spin3, incorrect unpaired length")

    terms = [
        (-0.40824829046386307, paired.copy() + [unpaired[1], unpaired[2], unpaired[3]], paired.copy() + [unpaired[0]]),
        (0.40824829046386307, paired.copy() + [unpaired[0], unpaired[2], unpaired[3]], paired.copy() + [unpaired[1]]),
        (0.81649658092772615, paired.copy() + [unpaired[0], unpaired[1], unpaired[3]], paired.copy() + [unpaired[2]])
    ]
    return CSF(3, terms)

def ne4spin3_c(paired, unpaired):
    if len(unpaired) != 4:
        raise ValueError("error in ne4spin3, incorrect unpaired length")

    terms = [
        (-0.28867513459481292, paired.copy() + [unpaired[1], unpaired[2], unpaired[3]], paired.copy() + [unpaired[0]]),
        (0.28867513459481292, paired.copy() + [unpaired[0], unpaired[2], unpaired[3]], paired.copy() + [unpaired[1]]),
        (-0.28867513459481292, paired.copy() + [unpaired[0], unpaired[1], unpaired[3]], paired.copy() + [unpaired[2]]),
        (-0.866025403784439, paired.copy() + [unpaired[0], unpaired[1], unpaired[2]], paired.copy() + [unpaired[3]])
    ]
    return CSF(4, terms)

def getCSF(spin, paired, unpaired):
    if spin == 1 and len(unpaired) == 0:
        return [ne0spin1(paired, unpaired)]
    elif spin == 2 and len(unpaired) == 1:
        return [ne1spin2(paired, unpaired)]
    elif spin == 1 and len(unpaired) == 2:
        return [ne2spin1(paired, unpaired)]
    elif spin == 3 and len(unpaired) == 2:
        return [ne2spin3(paired, unpaired)]
    elif spin == 2 and len(unpaired) == 3:
        c1 = ne3spin2_from_singlet(paired, unpaired)
        c2 = ne3spin2_from_triplet(paired, unpaired)
        return [c1,c2]
    elif spin == 4 and len(unpaired) == 3:
        raise NotImplementedError
        #return [ne3spin4(paired, unpaired)]
    elif spin == 1 and len(unpaired) == 4:
        #raise NotImplementedError
        c1 = ne4spin1_a(paired, unpaired)
        c2 = ne4spin1_b(paired, unpaired)
        return [c1, c2]
    elif spin == 3 and len(unpaired) == 4:
        #raise NotImplementedError
        c1 = ne4spin3_a(paired, unpaired)
        c2 = ne4spin3_b(paired, unpaired)
        c3 = ne4spin3_c(paired, unpaired)
        return [c1, c2, c3]
    else:
        raise NotImplementedError

if __name__ == "__main__":
# Example test case
    csf = getCSF(1, [1, 2], [])
    print(csf)
    print(csf.nterms)
    print(csf.terms)

    csf2 = getCSF(1, [1, 2], [3, 4])
    print(csf2)
    print(csf.nterms)
    print(csf.terms)
