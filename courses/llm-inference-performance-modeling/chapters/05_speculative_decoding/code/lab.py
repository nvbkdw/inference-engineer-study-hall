"""Exact finite-vocabulary speculative reference with history-dependent distributions."""
import math
import random


def validate(p):
    if not p or any(x < 0 or not math.isfinite(x) for x in p) or not math.isclose(sum(p), 1, abs_tol=1e-9):
        raise ValueError('expected a normalized probability vector')


def sample(p, rng):
    validate(p)
    u, cumulative = rng.random(), 0.0
    for token, prob in enumerate(p):
        cumulative += prob
        if u < cumulative:
            return token
    return max(i for i, prob in enumerate(p) if prob > 0)


def residual(p, q):
    mass = [max(a-b, 0) for a, b in zip(p, q)]
    total = sum(mass)
    if total <= 0:
        raise ValueError('zero residual mass: rejection is impossible for equal distributions')
    return [x/total for x in mass]


def cycle(history, draft, target, k, rng, remaining, eos=None):
    """Returns committed outputs and number accepted. No neural cache in this oracle.

    draft/target(history) return actual transformed probability vectors.
    Real implementation batches target calls; this oracle evaluates sequentially.
    """
    if k < 1 or remaining < 0:
        raise ValueError('k >= 1 and remaining >= 0 required')
    committed, accepted = [], 0
    for _ in range(min(k, remaining)):
        prefix = list(history)+committed
        q, p = draft(prefix), target(prefix)
        validate(q)
        validate(p)
        if len(p) != len(q):
            raise ValueError('vocabulary mismatch')
        x = sample(q, rng)
        if rng.random() < min(1, p[x]/q[x]):
            committed.append(x)
            accepted += 1
            if x == eos:
                return committed, accepted
        else:
            committed.append(sample(residual(p, q), rng))
            return committed, accepted
    if len(committed) < remaining:
        committed.append(sample(target(list(history)+committed), rng))
    return committed, accepted


def output_distribution(p, q):
    """Exact one-position enumeration: accepted mass + rejection correction mass."""
    validate(p)
    validate(q)
    accepted = [min(a, b) for a, b in zip(p, q)]
    reject = 1-sum(accepted)
    if reject < 1e-15:
        return accepted
    r = residual(p, q)
    return [a+reject*b for a, b in zip(accepted, r)]


def speedup(alpha, k, draft_ms, verify_ms, overhead_ms, target_ms):
    # Constant independent acceptance exercise; replace with measured E[A].
    return sum(alpha**j for j in range(k+1))*target_ms/(draft_ms+verify_ms+overhead_ms)


def check():
    for p, q in [([.7,.2,.1], [.2,.6,.2]), ([1.,0.,0.], [0.,1.,0.]), ([.5,.5], [.5,.5])]:
        assert all(abs(a-b) < 1e-12 for a,b in zip(output_distribution(p,q),p))
    rng = random.Random(42)
    p, q = [.7,.2,.1], [.2,.6,.2]
    counts = [0]*3
    n = 30000
    for _ in range(n):
        out, _ = cycle([], lambda h:q, lambda h:p, 4, rng, 1)
        counts[out[0]] += 1
    # Hoeffding union bound across three bins, failure probability <= 0.001.
    bound = math.sqrt(math.log(2*len(p)/.001)/(2*n))
    assert max(abs(c/n-prob) for c, prob in zip(counts,p)) < bound
    assert cycle([], lambda h:[1.,0.], lambda h:[1.,0.], 4, rng, 9, eos=0) == ([0],1)
    assert cycle([], lambda h:[1.,0.], lambda h:[1.,0.], 4, rng, 0) == ([],0)
    print('PASS: exact mass, zero-support cases, empirical distribution, EOS, length limit')
    for alpha in (.5,.8):
        print(f'ILLUSTRATIVE alpha={alpha}: speedup={speedup(alpha,4,1,1.2,.05,1):.3f}')


if __name__ == '__main__':
    check()
