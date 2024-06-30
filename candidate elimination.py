def more_general(h1, h2):
    """
    Check if hypothesis h1 is more general than hypothesis h2.
    """
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == '?' or (x != 'ϕ' and (x == y or y == 'ϕ'))
        more_general_parts.append(mg)
    return all(more_general_parts)

def satisfies(example, hypothesis):
    """
    Check if the hypothesis satisfies the given example.
    """
    return all(h == '?' or h == e for h, e in zip(hypothesis, example))

def min_generalizations(h, example):
    """
    Return the minimal generalizations of h that satisfy example.
    """
    h_new = list(h)
    for i, val in enumerate(h):
        if not satisfies([val], [example[i]]):
            h_new[i] = '?' if val != 'ϕ' else example[i]
    return [tuple(h_new)]

def min_specializations(h, domains, example):
    """
    Return the minimal specializations of h that satisfy example.
    """
    results = []
    for i, val in enumerate(h):
        if val == '?':
            for val_d in domains[i]:
                if example[i] != val_d:
                    h_new = list(h)
                    h_new[i] = val_d
                    results.append(tuple(h_new))
        elif val != 'ϕ':
            h_new = list(h)
            h_new[i] = 'ϕ'
            results.append(tuple(h_new))
    return results

def candidate_elimination(examples):
    """
    Candidate-Elimination Algorithm
    """
    domains = [set() for i in range(len(examples[0]) - 1)]
    for example in examples:
        for i, val in enumerate(example[:-1]):
            domains[i].add(val)

    S = [('ϕ',) * (len(examples[0]) - 1)]
    G = [('?',) * (len(examples[0]) - 1)]

    for example in examples:
        inputs, output = example[:-1], example[-1]

        if output == 'Yes':
            # Remove from G any hypothesis inconsistent with example
            G = [g for g in G if satisfies(inputs, g)]
            # For each hypothesis s in S that is not consistent with example,
            # remove s from S and add to S all minimal generalizations of s
            for s in S[:]:
                if not satisfies(inputs, s):
                    S.remove(s)
                    S += min_generalizations(s, inputs)
            # Remove from S any hypothesis that is more general than any other hypothesis in S
            S = [s for s in S if not any(more_general(s2, s) for s2 in S if s != s2)]

        else:
            # Remove from S any hypothesis inconsistent with example
            S = [s for s in S if not satisfies(inputs, s)]
            # For each hypothesis g in G that is not consistent with example,
            # remove g from G and add to G all minimal specializations of g
            for g in G[:]:
                if satisfies(inputs, g):
                    G.remove(g)
                    G += min_specializations(g, domains, inputs)
            # Remove from G any hypothesis that is more specific than any other hypothesis in G
            G = [g for g in G if not any(more_general(g, g2) for g2 in G if g != g2)]

    return S, G

# Dataset: Each example is a tuple of the form (attribute1, attribute2, attribute3, attribute4, label)
data = [
    ('Sunny', 'Warm', 'Normal', 'Strong', 'Yes'),
    ('Sunny', 'Warm', 'High', 'Strong', 'Yes'),
    ('Rainy', 'Cold', 'High', 'Strong', 'No'),
    ('Sunny', 'Warm', 'High', 'Strong', 'Yes')
]

# Running the Candidate-Elimination algorithm
S, G = candidate_elimination(data)

print("S (Specific Hypotheses):")
for s in S:
    print(s)

print("\nG (General Hypotheses):")
for g in G:
    print(g)
