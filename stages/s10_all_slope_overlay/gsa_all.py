"""All slopes for the overlay with the landslide geologically sensitive areas: sections S10 of an27_main.py and R4 of an27_rep.py run with the slope filter removed and bootstrap labels with the suffix ':all', after the run on slopes of 39.5 degrees or less has reproduced the registered values. Writes gsa_all_v1.json."""
import json, os, sys, time
H = '@WORK@'
V27 = os.path.join(H, 'v27', 'code'); sys.path.insert(0, V27)
os.environ['TAGS'] = 'v5'; os.environ['BOOT'] = '1'
T0 = time.time()
def say(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
def preamble(path, stop):
    src = open(path, encoding='utf-8').read().split('\n'); i = next(k for k, l in enumerate(src) if l.startswith(stop)); return '\n'.join(src[:i])
def block(path, start, end):
    src = open(path, encoding='utf-8').read(); a = src.index(start); b = src.index(end, a); return src[a:b]
def same(a, b, path=''):
    if isinstance(a, dict):
        assert set(a) == set(b), (path, sorted(set(a) ^ set(b))); [same(a[k], b[k], f'{path}/{k}') for k in a]
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path; [same(x, y, f'{path}[{i}]') for i, (x, y) in enumerate(zip(a, b))]
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)), (path, a, b)
    else: assert a == b, (path, a, b)
    return True
MAIN = os.path.join(V27, 'an27_main.py'); REPS = os.path.join(V27, 'an27_rep.py')
NM = {'__name__': 'an27_main_pre', '__file__': MAIN}
exec(compile(preamble(MAIN, 'def s0():'), 'an27_main_pre', 'exec'), NM)
S10SRC = block(MAIN, "ZF = H + '", 'S10 = cached(')
assert S10SRC.count('L = M[M.slope <= 39.5].copy()') == 1 and S10SRC.count("'gsa_model'") == 1 and S10SRC.count("'gsa_af_out'") == 1
S10SRC = (S10SRC.replace('def s10():', 'def s10(DOM, SUF):').replace('L = M[M.slope <= 39.5].copy()', 'L = M[DOM(M)].copy()')
          .replace("'gsa_model'", "'gsa_model' + SUF").replace("'gsa_af_out'", "'gsa_af_out' + SUF"))
exec(compile(S10SRC, 's10_all', 'exec'), NM)
REG10 = json.load(open(os.path.join(H, 'reference', 'main27_v5.json'), encoding='utf-8'))['S10']
res = {'S10': {}, 'R4': {}, 'validation': {}}
say('S10, slopes of 39.5 degrees or less, registered labels')
r = NM['clean'](NM['s10'](lambda d: d.slope <= 39.5, '')); res['validation']['S10_le39.5_equals_register'] = same(REG10, r); res['S10']['le39.5'] = r
say('S10, all slopes')
res['S10']['all'] = NM['clean'](NM['s10'](lambda d: d.slope >= 0, ':all'))
NR = {'__name__': 'an27_rep_pre', '__file__': REPS}
exec(compile(preamble(REPS, 'def period_block('), 'an27_rep_pre', 'exec'), NR)
R4SRC = block(REPS, 'def r4():', '    def strat_auc')
assert R4SRC.count('L = X[X.slope <= 39.5].copy()') == 1 and R4SRC.count("'rep_gsa'") == 1
R4SRC = R4SRC.replace('def r4():', 'def r4(DOM, SUF):').replace('L = X[X.slope <= 39.5].copy()', 'L = X[DOM(X)].copy()').replace("'rep_gsa'", "'rep_gsa' + SUF") + '    return o\n'
exec(compile('ZF = ' + repr(NM['ZF']) + '\n' + R4SRC, 'r4_all', 'exec'), NR)
REG4 = json.load(open(os.path.join(H, 'reference', 'rep27_v5.json'), encoding='utf-8'))['R4']['rep_legacy_OR_outside']
say('R4, slopes of 39.5 degrees or less, registered label')
r = NR['clean'](NR['r4'](lambda d: d.slope <= 39.5, ''))['rep_legacy_OR_outside']; res['validation']['R4_le39.5_equals_register'] = same(REG4, r); res['R4']['le39.5'] = r
say('R4, all slopes')
res['R4']['all'] = NR['clean'](NR['r4'](lambda d: d.slope >= 0, ':all'))['rep_legacy_OR_outside']
res['meta'] = {'boot': 2000, 'note': "le39.5 reproduces S10 of main27_v5.json and R4 of rep27_v5.json; all uses the bootstrap labels with the suffix ':all'"}
os.makedirs(os.path.join(H, 'v27', 'an'), exist_ok=True)
json.dump(res, open(os.path.join(H, 'v27', 'an', 'gsa_all_v1.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
say('saved gsa_all_v1.json')
