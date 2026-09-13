"""Bounded page ownership reference, independent of GPU tensor storage."""
import math


class Pool:
    def __init__(self, pages=8, page_size=4):
        if pages <= 0 or page_size <= 0:
            raise ValueError('positive capacity required')
        self.page_size = page_size
        self.refs = [0] * pages
        self.tables = {}
        self.lengths = {}

    def create(self, request):
        if request in self.tables:
            raise ValueError('duplicate request')
        self.tables[request], self.lengths[request] = [], 0

    def append(self, request, tokens):
        """Reserve atomically. Returns False on exhaustion; never partially mutates."""
        if tokens <= 0:
            raise ValueError('positive append required')
        old = self.lengths[request]
        table = self.tables[request]
        if old % self.page_size and self.refs[table[-1]] > 1:
            raise ValueError('shared partial blocks require copy-on-write')
        need = math.ceil((old+tokens)/self.page_size) - len(table)
        free = [i for i, count in enumerate(self.refs) if count == 0]
        if need > len(free):
            return False
        for page in free[:need]:
            self.refs[page] = 1
            table.append(page)
        self.lengths[request] += tokens
        self.check()
        return True

    def share_prefix(self, source, dest, tokens):
        if tokens < 0 or tokens % self.page_size or tokens > self.lengths[source]:
            raise ValueError('only complete, populated blocks may be shared')
        self.create(dest)
        self.tables[dest] = self.tables[source][:tokens // self.page_size].copy()
        self.lengths[dest] = tokens
        for page in self.tables[dest]:
            self.refs[page] += 1
        self.check()

    def release(self, request):
        for page in self.tables.pop(request):
            self.refs[page] -= 1
        del self.lengths[request]
        self.check()

    def check(self):
        expected = [0]*len(self.refs)
        for request, table in self.tables.items():
            assert len(table) == math.ceil(self.lengths[request]/self.page_size)
            assert len(table) == len(set(table))
            for page in table:
                expected[page] += 1
            if self.lengths[request] % self.page_size:
                assert self.refs[table[-1]] == 1
        assert expected == self.refs


def select_work(decode_ids, prefills, budget, chunk):
    """Token-budget selection only; caller must atomically reserve pages before run.

    prefills is [(request_id, remaining_tokens)]. One token per decode request.
    Caller implements fairness if decode demand exceeds the budget.
    """
    if budget < 1 or chunk < 1:
        raise ValueError('positive budget/chunk required')
    work = [(r, 1) for r in decode_ids[:budget]]
    remaining = budget-len(work)
    for r, tokens in prefills:
        take = min(tokens, chunk, remaining)
        if take > 0:
            work.append((r, take))
            remaining -= take
    return work


def check():
    pool = Pool(4, 4)
    pool.create('a')
    assert pool.append('a', 5)
    pool.share_prefix('a', 'b', 4)
    assert pool.append('b', 3)
    before = (pool.tables['a'].copy(), pool.lengths['a'])
    assert not pool.append('a', 20)
    assert before == (pool.tables['a'], pool.lengths['a'])
    pool.release('a')  # b must retain the shared prefix.
    assert pool.refs[pool.tables['b'][0]] == 1
    pool.release('b')
    assert not any(pool.refs)
    assert select_work(['d1', 'd2'], [('p', 10)], 6, 3) == [('d1', 1), ('d2', 1), ('p', 3)]
    print('PASS: ownership, prefix sharing, exhaustion atomicity, cancellation release')


if __name__ == '__main__':
    check()
