"""시장 그룹판 — 한국(업종 시총가중 지수)·미국·일본·유럽·중국 추세 보드를 index.html로 만든다.

실행:  python market_board.py           (매일 장 마감 후 1회)
       python market_board.py --cache   (오늘 받은 데이터를 재사용, 화면만 다시 그릴 때)
"""
import sys, html, pickle, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent
OUT = BASE / 'index.html'
CACHE = BASE / 'cache'
KR_MIN_CAP = 1e11          # 시총 1000억 원 이상만 업종 지수에 넣는다
KST = dt.timezone(dt.timedelta(hours=9))
MA_LINES = [(200, 'm200'), (120, 'm120'), (50, 'm50')]   # 차트에 그리는 일 이평선
MA = {'일': ('D', 50), '주': ('W', 30), '월': ('M', 10)}

# ── 해외 구성: (티커, 이름) ──────────────────────────────────────────────
SECTIONS = [
    dict(region='us', title='미국 — 지수·섹터·산업 ETF', bench='SPY',
         indices=[('SPY', 'S&P 500'), ('QQQ', '나스닥100'), ('IWM', '러셀2000'), ('RSP', 'S&P 500 동일가중')],
         items=[('XLK', '기술'), ('SMH', '반도체'), ('IGV', '소프트웨어'), ('XLC', '커뮤니케이션'),
                ('XLY', '경기소비재'), ('XRT', '소매'), ('XLP', '필수소비재'), ('XLV', '헬스케어'),
                ('XBI', '바이오'), ('IHI', '의료기기'), ('XLF', '금융'), ('KRE', '지방은행'),
                ('XLE', '에너지'), ('XOP', '석유·가스 개발'), ('XLI', '산업재'), ('ITA', '항공·방산'),
                ('IYT', '운송'), ('XLB', '소재'), ('GDX', '금광'), ('XME', '금속·광업'),
                ('XLU', '유틸리티'), ('XLRE', '부동산'), ('ITB', '주택건설'), ('ARKK', '혁신성장(투기 성향)'),
                ('TAN', '태양광'), ('URA', '우라늄')]),
    dict(region='jp', title='일본 — 지수·TOPIX-17 업종 ETF', bench='1306.T',
         indices=[('^N225', '닛케이225'), ('1306.T', 'TOPIX'), ('2516.T', '그로스250')],
         items=[('1617.T', '식품'), ('1618.T', '에너지자원'), ('1619.T', '건설·자재'), ('1620.T', '소재·화학'),
                ('1622.T', '자동차·수송기'), ('1623.T', '철강·비철'), ('1624.T', '기계'), ('1625.T', '전기·정밀'),
                ('1626.T', '정보통신·서비스'), ('1627.T', '전력·가스'), ('1628.T', '운수·물류'), ('1629.T', '상사·도매'),
                ('1630.T', '소매'), ('1631.T', '은행'), ('1632.T', '금융(은행 제외)'), ('1633.T', '부동산'),
                ('2644.T', '반도체')]),
    dict(region='eu', title='유럽 — 지수·STOXX 600 업종 ETF', bench='^STOXX',
         indices=[('^STOXX', 'STOXX 600'), ('^STOXX50E', '유로스톡스50'), ('^GDAXI', '독일 DAX'),
                  ('^FCHI', '프랑스 CAC40'), ('^FTSE', '영국 FTSE100')],
         items=[('EXV1.DE', '은행'), ('EXH2.DE', '금융서비스'), ('EXH5.DE', '보험'), ('EXV3.DE', '기술'),
                ('EXV2.DE', '통신'), ('EXH6.DE', '미디어'), ('EXV4.DE', '헬스케어'), ('EXV5.DE', '자동차·부품'),
                ('EXH7.DE', '생활용품'), ('EXH3.DE', '음식료'), ('EXH8.DE', '소매'), ('EXV9.DE', '여행·레저'),
                ('EXH4.DE', '산업재'), ('EXV8.DE', '건설·자재'), ('EXV7.DE', '화학'), ('EXV6.DE', '기초자원'),
                ('EXH1.DE', '석유·가스'), ('EXH9.DE', '유틸리티'), ('EXI5.DE', '부동산'), ('DFEN.DE', '방산(글로벌)')]),
    dict(region='cn', title='중국 — 미국 상장 ETF', bench='MCHI',
         indices=[('MCHI', 'MSCI 중국'), ('FXI', '중국 대형주50'), ('ASHR', 'CSI300 A주')],
         items=[('KWEB', '인터넷'), ('CQQQ', '기술'), ('KTEC', '항셍테크'), ('CHIQ', '경기소비재'),
                ('KURE', '헬스케어'), ('KGRN', '클린테크'), ('CNXT', '창업판(ChiNext)'), ('CNYA', 'MSCI A주'),
                ('PGJ', '나스닥 골든드래곤')]),
    dict(region='cn', title='중국 — 홍콩 상장 지수·ETF', bench='^HSI',
         indices=[('^HSI', '항셍'), ('^HSCE', '항셍중국기업(H주)'), ('3033.HK', '항셍테크'), ('000001.SS', '상하이종합')],
         items=[('3188.HK', 'CSI300'), ('2822.HK', 'FTSE A50'), ('3147.HK', '창업판(ChiNext)'),
                ('3191.HK', '반도체'), ('2826.HK', '클라우드'), ('2845.HK', '전기차·배터리'),
                ('2809.HK', '클린에너지'), ('2820.HK', '바이오'), ('3069.HK', '항셍바이오'), ('3110.HK', '항셍고배당')]),
]
KR_INDICES = [('^KS11', '코스피'), ('^KQ11', '코스닥')]
REGIONS = [('kr', '한국'), ('us', '미국'), ('jp', '일본'), ('eu', '유럽'), ('cn', '중국')]

# ── 한국 업종: KSIC 업종 → 그룹 ─────────────────────────────────────────
KR_GROUPS = {
    '반도체': ['반도체 제조업'],
    'IT하드웨어·전자부품': ['전자부품 제조업', '통신 및 방송 장비 제조업', '컴퓨터 및 주변장치 제조업', '영상 및 음향기기 제조업',
                       '측정, 시험, 항해, 제어 및 기타 정밀기기 제조업; 광학기기 제외', '사진장비 및 광학기기 제조업',
                       '마그네틱 및 광학 매체 제조업', '전구 및 조명장치 제조업'],
    '2차전지·전기장비': ['일차전지 및 이차전지 제조업', '전동기, 발전기 및 전기 변환 · 공급 · 제어 장치 제조업',
                    '절연선 및 케이블 제조업', '기타 전기장비 제조업'],
    '자동차·부품': ['자동차용 엔진 및 자동차 제조업', '자동차 신품 부품 제조업', '고무제품 제조업', '자동차 차체나 트레일러 제조업',
                '자동차 판매업', '자동차 부품 및 내장품 판매업'],
    '조선': ['선박 및 보트 건조업'],
    '방산·우주항공': ['항공기,우주선 및 부품 제조업', '무기 및 총포탄 제조업', '그외 기타 운송장비 제조업'],
    '기계·산업재': ['특수 목적용 기계 제조업', '일반 목적용 기계 제조업', '구조용 금속제품, 탱크 및 증기발생기 제조업',
               '기타 금속 가공제품 제조업', '기계장비 및 관련 물품 도매업', '산업용 기계 및 장비 임대업'],
    '건설·건자재': ['토목 건설업', '건물 건설업', '건축기술, 엔지니어링 및 관련 기술 서비스업', '시멘트, 석회, 플라스터 및 그 제품 제조업',
               '기타 비금속 광물제품 제조업', '유리 및 유리제품 제조업', '내화, 비내화 요업제품 제조업', '전기 및 통신 공사업',
               '실내건축 및 건축마무리 공사업', '기반조성 및 시설물 축조관련 전문공사업', '건물설비 설치 공사업', '나무제품 제조업'],
    '화학·소재': ['기초 화학물질 제조업', '기타 화학제품 제조업', '합성고무 및 플라스틱 물질 제조업', '화학섬유 제조업',
              '플라스틱제품 제조업', '비료, 농약 및 살균, 살충제 제조업', '펄프, 종이 및 판지 제조업',
              '골판지, 종이 상자 및 종이용기 제조업', '기타 종이 및 판지 제품 제조업'],
    '철강·비철금속': ['1차 철강 제조업', '1차 비철금속 제조업'],
    '에너지·유틸리티': ['석유 정제품 제조업', '전기업', '연료용 가스 제조 및 배관공급업', '증기, 냉·온수 및 공기조절 공급업',
                  '연료 소매업', '폐기물 처리업'],
    '바이오·헬스케어': ['기초 의약물질 제조업', '의약품 제조업', '의료용품 및 기타 의약 관련제품 제조업', '의료용 기기 제조업',
                  '자연과학 및 공학 연구개발업', '기타 과학기술 서비스업'],
    '인터넷·소프트웨어': ['소프트웨어 개발 및 공급업', '자료처리, 호스팅, 포털 및 기타 인터넷 정보매개 서비스업',
                   '컴퓨터 프로그래밍, 시스템 통합 및 관리업', '기타 정보 서비스업'],
    '미디어·엔터·게임': ['오디오물 출판 및 원판 녹음업', '영화, 비디오물, 방송프로그램 제작 및 배급업', '텔레비전 방송업', '광고업',
                   '창작 및 예술관련 서비스업'],
    '통신': ['전기 통신업'],
    '화장품·의류·생활용품': ['봉제의복 제조업', '가죽, 가방 및 유사제품 제조업', '신발 및 신발 부분품 제조업', '방적 및 가공사 제조업',
                     '직물직조 및 직물제품 제조업', '생활용품 도매업', '가정용 기기 제조업', '가구 제조업', '그외 기타 제품 제조업',
                     '그외 기타 개인 서비스업', '개인 및 가정용품 임대업', '개인 및 가정용품 수리업', '상품 종합 도매업'],
    '음식료·유통': ['기타 식품 제조업', '담배 제조업', '알코올음료 제조업', '비알코올음료 및 얼음 제조업', '동·식물성 유지 및 낙농제품 제조업',
               '동물용 사료 및 조제식품 제조업', '곡물가공품, 전분 및 전분제품 제조업', '수산물 가공 및 저장 처리업',
               '도축, 육류 가공 및 저장 처리업', '어로 어업', '작물 재배업', '음·식료품 및 담배 도매업', '종합 소매업',
               '기타 상품 전문 소매업', '무점포 소매업', '섬유, 의복, 신발 및 가죽제품 소매업', '가전제품 및 정보통신장비 소매업'],
    '운송': ['해상 운송업', '항공 여객 운송업', '기타 운송관련 서비스업', '도로 화물 운송업', '육상 여객 운송업', '운송장비 임대업'],
    '레저·교육·서비스': ['유원지 및 기타 오락관련 서비스업', '여행사 및 기타 여행보조 서비스업', '일반 및 생활 숙박시설 운영업',
                   '일반 교습 학원', '기타 교육기관', '서적, 잡지 및 기타 인쇄물 출판업', '경비, 경호 및 탐정업',
                   '기타 사업지원 서비스업', '기타 전문 서비스업', '전문디자인업', '그외 기타 전문, 과학 및 기술 서비스업'],
    '은행·금융지주': ['은행 및 저축기관'],
    '증권': ['금융 지원 서비스업', '신탁업 및 집합투자업'],
    '보험': ['보험업', '재 보험업', '보험 및 연금관련 서비스업'],
    '지주·상사': ['기타 금융업', '회사 본부 및 경영 컨설팅 서비스업', '기타 전문 도매업', '상품 중개업'],
    '부동산·리츠': ['부동산 임대 및 공급업'],
}
FINANCE = {'은행·금융지주', '증권', '보험', '지주·상사'}
# 주요제품 키워드로 다시 분류: (그룹, 키워드, 이 그룹에서만 옮김 — None이면 금융 외 전부)
KR_KEYWORDS = [
    ('반도체', ('반도체',), None),
    ('2차전지·전기장비', ('이차전지', '2차전지', '양극재', '음극재', '전해액', '분리막', '배터리'),
     {'화학·소재', '기계·산업재', 'IT하드웨어·전자부품', '철강·비철금속'}),
    ('화장품·의류·생활용품', ('화장품',), {'화학·소재', '음식료·유통'}),
    ('미디어·엔터·게임', ('게임',), {'인터넷·소프트웨어'}),
]
KR_OVERRIDE = {'005930': '반도체', '105560': '은행·금융지주', '055550': '은행·금융지주', '086790': '은행·금융지주',
               '316140': '은행·금융지주', '138930': '은행·금융지주', '175330': '은행·금융지주', '139130': '은행·금융지주',
               '138040': '은행·금융지주', '071050': '증권'}


# ── 데이터 ──────────────────────────────────────────────────────────────
def download(tickers, chunk=300):
    frames = []
    for i in range(0, len(tickers), chunk):
        part = yf.download(tickers[i:i + chunk], period='3y', auto_adjust=True, progress=False, threads=True)['Close']
        frames.append(part)
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated()].sort_index().replace(0, np.nan)


def kr_universe():
    lst = fdr.StockListing('KRX')[['Code', 'Name', 'Market', 'Marcap', 'Stocks']]
    desc = fdr.StockListing('KRX-DESC')[['Code', 'Industry', 'Products']]
    u = lst.merge(desc, on='Code')
    u = u[(u.Marcap >= KR_MIN_CAP) & u.Market.isin(['KOSPI', 'KOSDAQ']) & u.Code.str.endswith('0')
          & ~u.Name.str.contains('스팩')].copy()
    ind2grp = {ind: g for g, inds in KR_GROUPS.items() for ind in inds}

    def classify(r):
        if r.Code in KR_OVERRIDE:
            return KR_OVERRIDE[r.Code]
        g = ind2grp.get(r.Industry)
        prod = r.Products if isinstance(r.Products, str) else ''
        if g and g not in FINANCE:
            for grp, kws, only in KR_KEYWORDS:
                if (only is None or g in only) and any(k in prod for k in kws):
                    return grp
        return g
    u['Group'] = u.apply(classify, axis=1)
    u = u.dropna(subset=['Group'])
    u['Ticker'] = u.Code + np.where(u.Market == 'KOSPI', '.KS', '.KQ')
    return u.sort_values('Marcap', ascending=False)


def cap_weighted(px, shares):
    """전일 시총 비중으로 일간 수익률을 이어 붙인 지수(100 시작)."""
    ret = px.pct_change(fill_method=None)
    ret = ret.where(ret.abs() <= 0.305)            # 상하한 ±30% 밖은 오류·신규상장으로 보고 제외
    w = (px.ffill() * shares).shift(1)
    ok = ret.notna() & w.notna()
    r = (ret.fillna(0) * w.where(ok, 0)).sum(axis=1) / w.where(ok, 0).sum(axis=1)
    return (1 + r.fillna(0)).cumprod() * 100


def load(use_cache):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f'{dt.datetime.now(KST):%Y%m%d}.pkl'
    if use_cache and f.exists():
        return pickle.loads(f.read_bytes())
    uni = kr_universe()
    overseas = sorted({t for s in SECTIONS for t, _ in s['indices'] + s['items']} | {t for t, _ in KR_INDICES})
    print(f'해외 {len(overseas)}개, 한국 {len(uni)}종목 다운로드…')
    data = dict(ov=download(overseas), kr=download(uni.Ticker.tolist()), uni=uni)
    f.write_bytes(pickle.dumps(data))
    return data


# ── 지표 ────────────────────────────────────────────────────────────────
def resample(s, rule):
    return s if rule == 'D' else s.resample('W-FRI' if rule == 'W' else 'ME').last().dropna()


def trend_state(s, rule, n):
    s = resample(s, rule)
    ma = s.rolling(n).mean()
    if ma.dropna().shape[0] < 2:
        return None
    above, rising = s.iloc[-1] > ma.iloc[-1], ma.iloc[-1] > ma.iloc[-2]
    kind = 'up' if above and rising else 'down' if not above and not rising else 'mix'
    return kind, above


def ret_since(s, **offset):
    base = s.asof(s.index[-1] - pd.DateOffset(**offset))
    return (s.iloc[-1] / base - 1) * 100 if pd.notna(base) else np.nan


def stats(s, bench=None):
    s = s.dropna()
    if len(s) < 260:
        return None
    badges = []
    for label, (rule, n) in MA.items():
        now, prev = trend_state(s, rule, n), trend_state(s.iloc[:-1], rule, n)
        if now:
            badges.append((label, now[0], now[1], now != prev))
    r3 = ret_since(s, months=3)
    return dict(
        last=s.iloc[-1], date=s.index[-1], day=(s.iloc[-1] / s.iloc[-2] - 1) * 100,
        m1=ret_since(s, months=1), rel3=r3 - ret_since(bench.dropna(), months=3) if bench is not None else np.nan,
        dd52=(s.iloc[-1] / s[s.index > s.index[-1] - pd.Timedelta(weeks=52)].max() - 1) * 100,
        badges=badges, daily=s)


# ── 그리기 ──────────────────────────────────────────────────────────────
def pct(v, digits=0, unit='%'):
    if pd.isna(v):
        return '–'
    v = round(v, digits)
    return f'{v:+.{digits}f}{unit}'.replace('+0%', '0%').replace('-0%', '0%').replace('+0p', '0p').replace('-0p', '0p')


def spark(s):
    W, H, L, R, T, B = 300, 112, 6, 246, 12, 88
    w = resample(s, 'W').iloc[-104:]
    mas = [(cls, resample(s.rolling(n).mean(), 'W').reindex(w.index)) for n, cls in MA_LINES]
    base = w.iloc[0]
    vals = pd.concat([w] + [m for _, m in mas])
    lo, hi = vals.min(), vals.max()
    span = (hi - lo) or 1
    x = lambda i: L + (R - L) * i / (len(w) - 1)
    y = lambda v: B - (B - T) * (v - lo) / span
    line = ' '.join(f'{x(i):.1f},{y(v):.1f}' for i, v in enumerate(w))
    mlines = ''.join(f'<polyline class="ma {cls}" points="'
                     + ' '.join(f'{x(i):.1f},{y(v):.1f}' for i, v in enumerate(m) if pd.notna(v)) + '"/>'
                     for cls, m in mas)
    end = (w.iloc[-1] / base - 1) * 100
    peak = (w.max() / base - 1) * 100
    ey = min(max(y(w.iloc[-1]) + 4, T + 14), B + 2)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="sp">',
             f'{mlines}<polyline class="px" points="{line}"/>',
             f'<text x="{R + 4}" y="{ey:.1f}" class="lb end">{pct(end)}</text>']
    if peak - end >= 5:
        parts.append(f'<text x="{W - 2}" y="{T - 2}" class="lb pk" text-anchor="end">{pct(peak)}</text>')
    seen = set()
    for i, d in enumerate(w.index):
        if d.month in (1, 7) and (d.year, d.month) not in seen and d.day <= 7:
            seen.add((d.year, d.month))
            parts.append(f'<text x="{x(i):.1f}" y="{H - 6}" class="ax">{d:%y.%m}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def card(name, sub, st, kind, tip=''):
    changed = any(b[3] for b in st['badges'])
    cls = 'card' + (' idx' if kind == 'index' else ' chg' if changed else '')
    day_cls = 'up' if st['day'] > 0 else 'down' if st['day'] < 0 else ''
    badges = ''.join(
        f'<span class="b {k}{" now" if c else ""}">{lab}{"▲" if above else "▼"}</span>'
        for lab, k, above, c in st['badges'])
    foot = (f'1개월 {pct(st["m1"])} · 52주 고점 대비 {pct(st["dd52"])}' if kind == 'index'
            else f'1개월 {pct(st["m1"])} · 3개월 상대 {pct(st["rel3"], unit="%p")}')
    t = f' title="{html.escape(tip)}"' if tip else ''
    return (f'<div class="{cls}"{t}><div class="hd"><b>{html.escape(name)}</b><span class="tk">{html.escape(sub)}</span>'
            f'<span class="d {day_cls}">{pct(st["day"], 1)}</span></div>{spark(st["daily"])}'
            f'<div class="bd">{badges}</div><div class="ft">{foot}</div></div>')


def section(sid, title, index_cards, items):
    items = sorted(items, key=lambda c: -np.nan_to_num(c[0], nan=-1e9))
    body = ''.join(index_cards) + ''.join(c for _, c in items)
    return f'<section id="{sid}"><h2>{title}</h2><div class="grid">{body}</div></section>'


def build(data):
    ov, kr, uni = data['ov'], data['kr'], data['uni']
    out, asof = [], {}

    # 한국: 지수 + 업종별 시총가중 지수
    kospi = ov['^KS11'].dropna()
    kr = kr[kr.notna().sum(axis=1) > kr.shape[1] * 0.5]      # 휴장일 행 제거
    idx_cards = [card(nm, '', st, 'index') for t, nm in KR_INDICES if (st := stats(ov[t]))]
    items = []
    for g, members in uni.groupby('Group'):
        cols = [t for t in members.Ticker if t in kr and kr[t].notna().sum() > 20]
        if len(cols) < 3:
            continue
        shares = members.set_index('Ticker').loc[cols, 'Stocks']
        st = stats(cap_weighted(kr[cols], shares), kospi)
        if st:
            top = ', '.join(members.Name.head(6))
            items.append((st['rel3'], card(g, f'{len(cols)}종목', st, 'item', f'시총 상위: {top}')))
    asof['kr'] = kospi.index[-1]
    out.append(section('kr', '한국 — 업종은 구성종목 시총 가중 지수', idx_cards, items))

    # 해외
    for s in SECTIONS:
        bench = ov[s['bench']]
        idx_cards = [card(nm, t.lstrip('^'), st, 'index') for t, nm in s['indices'] if t in ov and (st := stats(ov[t]))]
        items = [(st['rel3'], card(nm, t.lstrip('^'), st, 'item'))
                 for t, nm in s['items'] if t in ov and (st := stats(ov[t], bench))]
        sid = s['region'] + ('-2' if s['region'] in asof else '')
        asof[s['region']] = max(asof.get(s['region'], pd.Timestamp(0)), bench.dropna().index[-1])
        out.append(section(sid, s['title'], idx_cards, items))

    nav = ' '.join(f'<a href="#{rid}">{nm} <small>{asof[rid]:%m.%d}</small></a>' for rid, nm in REGIONS if rid in asof)
    return PAGE.format(nav=nav, body=''.join(out), now=dt.datetime.now(KST).strftime('%Y-%m-%d %H:%M KST'))


PAGE = '''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>시장 그룹판</title>
<style>
:root{{--bg:#f6f6f4;--card:#fff;--line:#e3e3df;--ink:#1d1d1f;--sub:#86868b;--up:#d6303a;--down:#2563c9;
--ma:#e8742f;--m120:#2e9e6a;--m200:#8b5cf6;--upbg:#fdecec;--downbg:#e8f0fc;--mixbg:#eeeeec;--chg:#d96a2b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151516;--card:#1f1f21;--line:#333336;--ink:#ececec;--sub:#9a9aa0;
--up:#ff5b62;--down:#5b93ff;--upbg:#3a1f22;--downbg:#1d2a40;--mixbg:#2c2c2f}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font-family:Pretendard,"Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;font-size:14px}}
header{{padding:20px 16px 8px;max-width:1280px;margin:auto}}h1{{font-size:24px;margin:0 0 6px}}
h1 small{{font-size:13px;color:var(--sub);font-weight:400;margin-left:6px}}
.tip{{color:var(--sub);font-size:12.5px;line-height:1.8}}
nav{{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;
display:flex;gap:6px;flex-wrap:wrap;justify-content:center}}
nav a{{color:var(--ink);text-decoration:none;padding:5px 12px;border-radius:999px;background:var(--card);border:1px solid var(--line);font-weight:600}}
nav small{{color:var(--sub);font-weight:400}}
main{{max-width:1280px;margin:auto;padding:0 16px 40px}}section{{scroll-margin-top:56px}}
h2{{font-size:18px;margin:26px 0 12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 12px 10px}}
.card.idx{{border:1.5px solid var(--ink)}}.card.chg{{border:1.5px solid var(--chg)}}
.hd{{display:flex;align-items:baseline;gap:6px}}.hd b{{font-size:15px}}
.tk{{color:var(--sub);font-size:12px}}.d{{margin-left:auto;font-weight:700}}
.up{{color:var(--up)}}.down{{color:var(--down)}}
.sp{{width:100%;height:auto;display:block;margin:4px 0 2px}}
.px{{fill:none;stroke:var(--ink);stroke-width:1.4;stroke-linejoin:round}}.ma{{fill:none;stroke-width:1.2;opacity:.9}}.m50{{stroke:var(--ma)}}.m120{{stroke:var(--m120)}}.m200{{stroke:var(--m200)}}
.c50{{color:var(--ma)}}.c120{{color:var(--m120)}}.c200{{color:var(--m200)}}
.lb{{font-size:11px;font-weight:700;fill:var(--ink)}}.lb.pk{{font-weight:400;fill:var(--sub)}}
.ax{{font-size:10px;fill:var(--sub)}}
.bd{{display:flex;gap:4px}}.b{{font-size:11.5px;padding:1px 5px;border-radius:5px;font-weight:600;border:1.5px solid transparent}}
.b.up{{background:var(--upbg);color:var(--up)}}.b.down{{background:var(--downbg);color:var(--down)}}
.b.mix{{background:var(--mixbg);color:var(--sub)}}.b.now{{border-color:var(--chg)}}
.ft{{text-align:right;color:var(--sub);font-size:12px;margin-top:4px}}
footer{{text-align:center;color:var(--sub);font-size:12px;padding:20px}}
</style></head><body>
<header><h1>시장 그룹판<small>일·주·월 추세 · 2년 주봉 종가와 50·120·200일선</small></h1>
<div class="tip">칩: <span class="b up">일▲</span> 이평 위·이평 상승 &nbsp;<span class="b down">주▼</span> 이평 아래·이평 하락
&nbsp;<span class="b mix">월▲</span> 엇갈림 &nbsp;<span class="b up now">일▲</span> 오늘 바뀜 · 일=50일선, 주=30주선, 월=10개월선.
선: <b class="c50">━ 50일</b> <b class="c120">━ 120일</b> <b class="c200">━ 200일</b>.<br>업종은 3개월 상대수익률(지수 대비) 순. 지수 카드는 검은 테, 오늘 칩이 바뀐 카드는 주황 테.</div></header>
<nav>{nav}</nav><main>{body}</main>
<footer>생성 {now} · 데이터: Yahoo Finance, KRX(FinanceDataReader) · 한국 업종은 시총 1000억 이상 종목, 현재 상장주식수 기준 근사</footer>
</body></html>'''


if __name__ == '__main__':
    data = load('--cache' in sys.argv)
    OUT.write_text(build(data), encoding='utf-8')
    print('저장:', OUT)
