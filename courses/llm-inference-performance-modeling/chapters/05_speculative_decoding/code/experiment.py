"""Estimate finite-vocabulary cycle outcomes and an explicitly invented cost model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from experiment import begin, finish, plot, write_csv, write_json
from lab import cycle
import math
import random


def target(history):
    shift=sum(history)%3
    base=[.7,.2,.1]
    return base[shift:]+base[:shift]


def main():
    args=begin('05',__doc__,['Finite vocabulary of three symbols; no pretrained model or performance measurement',
        'Distribution depends on sum of history; 2000 cycles per condition/repeat',
        'Illustrative times in target-step units: draft=.25*k, verify=1+.05*k, overhead=.05'], measurement=False)
    write_json(args.out/'prediction.json',dict(hypothesis='Higher accepted-prefix mass increases useful tokens/cycle; k can still cost more than it saves.',
                                              k=[1,2,4,8],draws_per_condition=2000))
    drafts={'exact':target,'uniform':lambda h:[1/3]*3,'misaligned':lambda h:list(reversed(target(h)))}
    rows,histograms=[],[]
    largest_deviation=0.
    # Family-wise Hoeffding bound over first-token bins for every run/condition.
    epsilon=math.sqrt(math.log(2*3*len(drafts)*4*args.repeats/.001)/(2*2000))
    for repeat in range(args.repeats):
        for name,draft in drafts.items():
            for k in [1,2,4,8]:
                rng=random.Random(args.seed+repeat)  # Matched starting stream; RNG usage differs by policy.
                accepted=[0]*(k+1)
                first=[0]*3
                useful=0
                for _ in range(2000):
                    out,count=cycle([],draft,target,k,rng,remaining=k+1)
                    first[out[0]]+=1
                    accepted[count]+=1
                    useful+=len(out)
                deviation=max(abs(count/2000-p) for count,p in zip(first,target([])))
                largest_deviation=max(largest_deviation,deviation)
                assert deviation<=epsilon, 'distribution check outside declared family-wise bound'
                emitted=useful/2000
                speedup=emitted/(.25*k+1+.05*k+.05)
                rows.append(dict(repeat=repeat,draft=name,k=k,empirical_mean_emitted=emitted,
                                 modeled_speedup=speedup,first_token_max_deviation=deviation))
                histograms.extend(dict(repeat=repeat,draft=name,k=k,accepted_prefix=j,cycles=count)
                                  for j,count in enumerate(accepted))
    write_csv(args.out/'accepted_prefix.csv',histograms)
    plot(args.out/'break_even.svg',rows,'k','modeled_speedup','draft','P5 invented cost model with sampled finite-vocabulary outcomes')
    finish(args,rows,dict(distribution_checks='passed',familywise_error_bound=.001,
                         hoeffding_epsilon=epsilon,max_first_token_deviation=largest_deviation,
                         scope='finite-vocabulary correctness and cost sensitivity; no Qwen speed claim'))


if __name__ == '__main__':
    main()
