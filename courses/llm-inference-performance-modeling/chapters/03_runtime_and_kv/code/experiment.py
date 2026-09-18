"""Deterministic serving simulation with real allocator logic and invented service times."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from experiment import begin, finish, plot, write_csv, write_json
from lab import Pool, select_work
import math
import random


def simulate(trace, chunk, pages=256, page_size=16, token_budget=256):
    pool = Pool(pages,page_size)
    waiting, active, done = [], {}, []
    time_ms, cursor, reserved, peak_pages = 0.,0,0,0
    while cursor < len(trace) or waiting or active:
        if not active and not waiting and cursor < len(trace):
            time_ms = max(time_ms,trace[cursor]['arrival_ms'])
        while cursor < len(trace) and trace[cursor]['arrival_ms'] <= time_ms:
            waiting.append(dict(trace[cursor],processed=0,emissions=[],state='prefill'))
            cursor += 1
        # Conservatively reserve future capacity at admission to avoid growth deadlock.
        while waiting:
            request = waiting[0]
            maximum = math.ceil((request['prompt']+request['output']-1)/page_size)
            if maximum > pages:
                raise ValueError('request cannot fit the configured pool')
            if reserved+maximum > pages:
                break
            waiting.pop(0)
            request['reservation'] = maximum
            reserved += maximum
            pool.create(request['id'])
            active[request['id']] = request
        if not active:
            raise RuntimeError('admission made no progress')
        decodes = [i for i,r in active.items() if r['state']=='decode']
        prefills = [(i,r['prompt']-r['processed']) for i,r in active.items() if r['state']=='prefill']
        work = select_work(decodes,prefills,token_budget,chunk)
        assert work
        prefill_tokens = sum(n for i,n in work if active[i]['state']=='prefill')
        decode_tokens = len([i for i,n in work if active[i]['state']=='decode'])
        # Sequential prefill + decode service. No hardware calibration is claimed.
        duration = .1 + .008*prefill_tokens + .00003*prefill_tokens**2 + .04*decode_tokens
        time_ms += duration
        for identity,tokens in work:
            r = active[identity]
            assert pool.append(identity,tokens), 'admission reservation must prevent exhaustion'
            r['processed'] += tokens
            if r['state']=='decode' or r['processed']==r['prompt']:
                r['state']='decode'
                r['emissions'].append(time_ms)
        peak_pages = max(peak_pages,sum(count>0 for count in pool.refs))
        for identity in list(active):
            r = active[identity]
            if len(r['emissions']) == r['output']:
                done.append(r)
                reserved -= r['reservation']
                pool.release(identity)
                del active[identity]
    assert not any(pool.refs) and reserved==0
    return done,peak_pages,time_ms-trace[0]['arrival_ms']


def main():
    args = begin('02',__doc__,['Every time/rate is simulated, never GPU measured',
        'Decode first, FCFS admission, token budget 256, conservative full-lifetime page reservation',
        'Illustrative fixed SLO: TTFT <= 20 ms and maximum token gap <= 5 ms'], measurement=False)
    write_json(args.out/'prediction.json',dict(hypothesis='Smaller chunks reduce prefill blocking but add iteration overhead.',
                                              chunks=[16,64,256],service_ms='.1+.008*P+.00003*P^2+.04*D'))
    rows,requests,traces=[],[],[]
    for repeat in range(args.repeats):
        rng = random.Random(args.seed+repeat)
        arrival=0.
        trace=[]
        for identity in range(90):
            arrival += rng.expovariate(1/1.0)
            trace.append(dict(id=identity,arrival_ms=arrival,prompt=256 if rng.random()<.2 else 32,output=8))
        traces.append(trace)
        for chunk in [16,64,256]:
            done,peak,duration = simulate(trace,chunk)
            eligible=0
            for r in done:
                ttft=r['emissions'][0]-r['arrival_ms']
                gap=max(b-a for a,b in zip(r['emissions'],r['emissions'][1:]))
                eligible += ttft<=20 and gap<=5
                requests.append(dict(repeat=repeat,chunk=chunk,request_id=r['id'],
                                     simulated_ttft_ms=ttft,simulated_max_gap_ms=gap))
            rows.append(dict(repeat=repeat,chunk=chunk,kind='simulation',completed=len(done),
                             peak_pages=peak,simulated_duration_ms=duration,
                             simulated_goodput_rps=1000*eligible/duration))
    write_json(args.out/'traces.json',traces)
    write_csv(args.out/'requests.csv',requests)
    plot(args.out/'goodput.svg',rows,'chunk','simulated_goodput_rps','kind','P2 simulated policy comparison: median and range')
    finish(args,rows,dict(scope='service model simulation',all_requests_completed=True,
                         ownership_released=True,trace_replays=args.repeats*3))


if __name__ == '__main__':
    main()
