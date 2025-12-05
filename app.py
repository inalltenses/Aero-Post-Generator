import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import torch

# ==========================================
# 1. 설정 및 데이터 로드 (캐싱 적용)
# ==========================================
st.set_page_config(page_title="Aero-Post Generator", page_icon="✈️", layout="wide")

@st.cache_resource
def load_resources():
    # 1. 모델 로드
    model_path = "./model"  # 로컬 폴더 경로
    try:
        model = BertForTokenClassification.from_pretrained(model_path)
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        
        # 라벨 정보 수동 주입 (config.json에 저장 안 됐을 경우 대비)
        id2label = {0: 'B-AIRCRAFT', 1: 'B-AIRLINE', 2: 'B-DATE', 3: 'I-ROUTE', 4: 'O'}
        label2id = {v: k for k, v in id2label.items()}
        model.config.id2label = id2label
        model.config.label2id = label2id
        
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, {}, {}, {}, {}, {}, {}

    # 2. 데이터 로드
    try:
        df_airport = pd.read_csv('data/airports_list.csv', encoding='cp949').dropna(subset=['공항코드1(IATA)', '한글공항'])
        df_airline = pd.read_csv('data/airlines_list.csv', encoding='cp949').dropna(subset=['항공사코드_IATA', '한글항공사명'])
        
        # 기종 데이터 로드 (인코딩 자동 감지 시도)
        try:
            df_aircraft = pd.read_csv('data/aircrafts_list.csv', encoding='utf-8')
        except:
            df_aircraft = pd.read_csv('data/aircrafts_list.csv', encoding='cp949')

        # 사전 구축
        airport_dict = dict(zip(df_airport['공항코드1(IATA)'], df_airport['한글공항']))
        airline_dict = dict(zip(df_airline['항공사코드_IATA'], df_airline['한글항공사명']))
        
        name_to_kor_airline = dict(zip(df_airline['영문항공사명'], df_airline['한글항공사명']))
        name_to_kor_airport = dict(zip(df_airport['영문공항명'], df_airport['한글공항']))
        name_to_kor_airport.update(dict(zip(df_airport['영문도시명'], df_airport['한글공항'])))
        
        # 기종 사전 (IATA -> FullName)
        aircraft_dict = {}
        for _, row in df_aircraft.iterrows():
            code = str(row['항공기코드_IATA']).strip()
            if code and code != 'nan':
                aircraft_dict[code] = (str(row['제조사']).title(), str(row['비행기모델']))

        # 수동 보정
        manual_fixes = {
            'MAD': '마드리드 바라하스 국제공항', 'DOH': '도하 하마드 국제공항',
            'TUN': '튀니스 카르타고 공항', 'BER': '베를린 브란덴부르크 공항',
            'ICN': '인천국제공항', 'NRT': '도쿄 나리타 공항',
            'HKG': '홍콩 첵랍콕 국제공항', 'SFO': '샌프란시스코 국제공항',
            'KUL': '쿠알라룸푸르 국제공항', 'CGK': '자카르타 수카르노 하타 국제공항',
            'HAN': '하노이 노이바이 국제공항', 'DAD': '다낭 국제공항',
            'OOL': '골드코스트(쿨랑가타) 공항',
            'Air Europa': '에어 유로파', 'Eurowings': '유로윙스',
            'Hong Kong Airlines': '홍콩항공', 'Asiana Airlines': '아시아나항공'
        }
        airport_dict.update(manual_fixes)
        airline_dict.update(manual_fixes)
        name_to_kor_airline.update(manual_fixes)

        # 검색용 리스트
        sorted_airline_names = sorted(name_to_kor_airline.keys(), key=len, reverse=True)
        sorted_airport_names = sorted(name_to_kor_airport.keys(), key=len, reverse=True)
        
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return nlp, {}, {}, {}, {}, {}, {}

    return nlp, airport_dict, airline_dict, name_to_kor_airline, name_to_kor_airport, sorted_airline_names, sorted_airport_names, aircraft_dict

# 리소스 로드
nlp, airport_dict, airline_dict, name_to_kor_airline, name_to_kor_airport, sorted_airline_names, sorted_airport_names, aircraft_dict = load_resources()

# ==========================================
# 2. 헬퍼 함수들 (Logic)
# ==========================================
def clean_garbage_words(text):
    if not text: return ""
    text = re.sub(r'(route|service|airport|flights?|eff)$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(route|service|airport|flights?|eff)\b', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def clean_location_name(name):
    if not name: return ""
    name = clean_garbage_words(name)
    name = re.sub(r'\s*\([^)]*\)$', '', name).strip()
    return name

def normalize_aircraft_name(name):
    if not name: return ""
    name = name.upper().replace("BOEING", "B").replace("AIRBUS", "A").strip()
    if re.match(r'^7\d{2}', name): name = "B" + name
    name = name.replace("B B", "B").replace("A A", "A")
    return name

def get_aircraft_fullname(code):
    if not code: return ""
    c = code.upper().replace("BOEING", "").replace("AIRBUS", "").replace(" ", "").replace("-", "")
    if c.startswith("B7"): c = c[1:]
    candidates = [c, "A"+c, "B"+c, c.replace("A", ""), c.replace("B", "")]
    for cand in candidates:
        if cand in aircraft_dict:
            maker, model = aircraft_dict[cand]
            maker_kr = "보잉" if "BOEING" in maker.upper() else "에어버스" if "AIRBUS" in maker.upper() else maker
            return f"{maker_kr} {model}"
    if "777" in code: return "보잉 777"
    if "787" in code: return "보잉 787"
    if "350" in code: return "에어버스 A350"
    if "330" in code: return "에어버스 A330"
    if "320" in code or "321" in code: return "에어버스 A320 계열"
    return code

def format_date(raw_date):
    try:
        months = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6, 'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
        match = re.match(r'(\d{1,2})([A-Z]{3})(\d{2})', raw_date)
        if match:
            day, m_str, _ = match.groups()
            return f"{months.get(m_str.upper())}월 {day}일"
    except: pass
    return ""

def format_time_pretty(time_str):
    match = re.match(r'(\d{2})(\d{2})([+]?\d*)', time_str)
    if match:
        hh, mm, plus = match.groups()
        time_fmt = f"{hh}:{mm}"
        if plus: time_fmt += f"({plus})"
        return time_fmt
    return time_str

def get_korean_smart(text, raw_text, type='airline'):
    text = clean_garbage_words(text)
    if type == 'airline' and text in airline_dict: return airline_dict[text]
    if type == 'airport' and text in airport_dict: return airport_dict[text]
    if type == 'airline' and text in name_to_kor_airline: return name_to_kor_airline[text]
    
    search_list = sorted_airline_names if type == 'airline' else sorted_airport_names
    target_dict = name_to_kor_airline if type == 'airline' else name_to_kor_airport
    
    found_candidate = None
    for name in search_list:
        if not name: continue
        if name in raw_text:
            kor_name = target_dict.get(name)
            if not kor_name: continue
            if type == 'airport':
                if '국제공항' in kor_name or 'International' in name: return kor_name
                if not found_candidate: found_candidate = kor_name
            else: return kor_name
    
    if len(text) > 2:
        for eng_name, kor_name in target_dict.items():
            if not eng_name: continue
            if text.lower() in str(eng_name).lower():
                if type == 'airport':
                    if '국제공항' in kor_name or 'International' in eng_name: return kor_name
                else: return kor_name
    if found_candidate: return found_candidate
    return text

def extract_frequency_and_days(text):
    if not text: return None, None
    t = text.lower()
    weekday_map = {"1": "월", "2": "화", "3": "수", "4": "목", "5": "금", "6": "토", "7": "일"}
    frequency, days_list = None, None
    weekly_patterns = [r'(\d+)\s*weekly', r'operates\s+(\d+)\s*weekly', r'(\d+)\s+times\s+weekly', r'(\d+)\s*×\s*weekly']
    for pat in weekly_patterns:
        m = re.search(pat, t)
        if m:
            try:
                val = int(m.group(1))
                if 1 <= val <= 30: frequency = str(val); break
            except: continue
    x_matches = re.findall(r'x([1-7]{1,7})', t)
    if x_matches:
        digits = x_matches[0]
        seen = set()
        days = []
        for d in digits:
            if d in weekday_map and d not in seen: seen.add(d); days.append(weekday_map[d])
        if days:
            days_list = days
            if frequency is None: frequency = str(len(days))
    return frequency, days_list

def extract_network_routes(text):
    routes = []
    lines = text.splitlines()
    pattern = re.compile(r"^\s*(?P<start>.+?)\s+[–-]\s+(?P<end>.+?)\s+eff\s+(?P<eff>\d{2}[A-Z]{3}\d{2})\s+(?P<count>\d+)\s+(?P<unit>weekly|daily)(?:\s+(?P<ac>.+?))?(?:\s*\(.*)?$", re.IGNORECASE)
    for line in lines:
        m = pattern.match(line.strip())
        if not m: continue
        start_en = clean_location_name(m.group("start"))
        end_en = clean_location_name(m.group("end"))
        start_en = re.sub(r'\s[A-Z]{2}$', '', start_en)
        end_en = re.sub(r'\s[A-Z]{2}$', '', end_en)
        routes.append({
            "start_en": start_en, "end_en": end_en, "eff_date": m.group("eff"),
            "count": m.group("count"), "unit": m.group("unit").lower(), "aircraft": (m.group("ac") or "").strip()
        })
    return routes

def classify_action_from_title(title, text, is_codeshare, aircraft_str):
    t = (title + " " + text[:200]).lower()
    if is_codeshare or "codeshare" in t: return "코드쉐어", "노선의 공동운항 협약을 맺었습니다."
    if any(k in t for k in ["launches", "launch", "inaugural", "new route", "plans", "opens new"]): return "신규 취항", "해당 노선을 신규 취항합니다."
    if any(k in t for k in ["resumes", "resume", "restores", "reinstates", "relaunches"]): return "운항 재개", "중단되었던 노선 운항을 재개합니다."
    if any(k in t for k in ["extra", "increase", "increases", "boosts"]): return "증편", "노선 운항을 증편합니다."
    if any(k in t for k in ["reduces", "reduce", "suspends", "suspension"]): return "감편/단축", "노선 운항을 감편합니다."
    aircraft_str = (aircraft_str or "").strip()
    if aircraft_str: return f"{aircraft_str} 투입", f"노선에 {aircraft_str} 기재를 투입(또는 증편)합니다."
    else: return "노선 변경", "노선 스케줄을 변경합니다."

def generate_caption(title, text, link):
    raw_text = text
    if "Published at" in text: text = text.split("Published at")[-1]
    text = re.sub(r'^.*GMT \d{2}[A-Z]{3}\d{2}', '', text).strip()
    
    results = nlp(text)
    info = {"AIRLINE": [], "AIRCRAFT": "", "DATE": "", "ROUTE_START": "", "ROUTE_END": ""}
    for entity in results:
        label, word = entity['entity_group'], entity['word'].replace(" ##", "").replace("##", "")
        if "AIRLINE" in label and word not in info['AIRLINE']: info['AIRLINE'].append(word)
        elif "AIRCRAFT" in label and not info['AIRCRAFT']: info['AIRCRAFT'] = word
        elif "DATE" in label and not info['DATE']: info['DATE'] = word
        
    is_codeshare = False
    if "codeshare" in text.lower() or "partner" in text.lower(): is_codeshare = True
    if "/" in text and re.search(r'[A-Z0-9]{2,3}\d{3,4}\/[A-Z0-9]{2,3}\d{3,4}', text): is_codeshare = True

    ac_match = re.search(r'(A3\d{2}(-\d{3,4})?|7\d{2}(-\d{3,4})?|Boeing\s7\d{2}|Airbus\sA3\d{2}|A220-300)', text)
    if ac_match: info['AIRCRAFT'] = normalize_aircraft_name(ac_match.group(0))
    else: info['AIRCRAFT'] = normalize_aircraft_name(info['AIRCRAFT'])
    
    date_match = re.search(r'\d{1,2}[A-Z]{3}\d{2}', text)
    if date_match: info['DATE'] = date_match.group(0)
    
    route_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s?[–-]\s?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', text)
    if route_match:
        info['ROUTE_START'] = clean_location_name(route_match.group(1))
        info['ROUTE_END'] = clean_location_name(route_match.group(2))

    schedule_pattern = r'([A-Z0-9]{2,3}\d{3,4}(?:/[A-Z0-9]{2,3}\d{3,4})?)\s+([A-Z]{3})(\d{3,4}[+]?\d*)\s*[–-]\s*(\d{3,4}(?:[+]?\d*)?)\s*([A-Z]{3})'
    schedules = re.findall(schedule_pattern, text)
    network_routes = extract_network_routes(text)
    frequency, days_list = extract_frequency_and_days(text)

    potential_airlines = info['AIRLINE'][:]
    for eng_name, kor_name in name_to_kor_airline.items():
        if eng_name in title and eng_name not in potential_airlines: potential_airlines.append(eng_name)
    
    seen, airlines_kr = set(), []
    for al in potential_airlines:
        kr = get_korean_smart(al, raw_text, 'airline')
        if kr not in seen and kr != al: seen.add(kr); airlines_kr.append(kr)
    if not airlines_kr: airlines_kr = ["해당 항공사"]
    airline_text = "-".join(airlines_kr[:2])

    if schedules:
        _, dep_code, _, _, arr_code = schedules[0]
        start_kr = airport_dict.get(dep_code, info['ROUTE_START'] or "출발지")
        end_kr = airport_dict.get(arr_code, info['ROUTE_END'] or "도착지")
    elif network_routes:
        start_kr = get_korean_smart(network_routes[0]['start_en'], raw_text, 'airport')
        end_kr = get_korean_smart(network_routes[0]['end_en'], raw_text, 'airport')
    else:
        start_kr = get_korean_smart(info['ROUTE_START'], raw_text, 'airport') if info['ROUTE_START'] else "출발지"
        end_kr = get_korean_smart(info['ROUTE_END'], raw_text, 'airport') if info['ROUTE_END'] else "도착지"

    date_kr = format_date(info['DATE'])
    title_suffix, body_msg = classify_action_from_title(title, text, is_codeshare, info['AIRCRAFT'])

    caption = []
    month_text = f"{date_kr.split('월')[0]}월" if date_kr and "월" in date_kr else ""
    headline = f"✈️ [뉴스] {airline_text}, {month_text}부터 {end_kr} {title_suffix}" if month_text else f"✈️ [뉴스] {airline_text}, {end_kr} {title_suffix}"
    caption.append(headline + "\n")
    
    start_msg = f"오는 {date_kr}부터" if date_kr else ""
    caption.append(f"📢 {airline_text}이(가) {start_msg} {start_kr} - {end_kr} {body_msg}")
    
    ac_full = get_aircraft_fullname(info['AIRCRAFT'])
    if ac_full and not is_codeshare:
        caption.append(f"💺 해당 노선에는 {ac_full}이(가) 투입될 예정입니다.")

    if schedules:
        freq_line = f"해당 운항편은 주 {frequency}회 편성되며 ({', '.join(days_list)})" if frequency and days_list and int(frequency) == len(days_list) else f"해당 운항편은 주 {frequency}회 편성되며" if frequency else "상세 스케줄은 다음과 같습니다"
        caption.append(f"\n🗓️ {freq_line}, 상세 스케줄은 다음과 같습니다:\n")
        for s in schedules:
            flt, dep_code, dep_tm, arr_tm, arr_code = s 
            if len(flt) >= 5 and flt[0].isdigit(): flt = flt[1:]
            dep_nm = airport_dict.get(dep_code, dep_code)
            arr_nm = airport_dict.get(arr_code, arr_code)
            caption.append(f"  * {flt}: {dep_nm}({dep_code}) {format_time_pretty(dep_tm)} 출발 ➔ {arr_nm}({arr_code}) {format_time_pretty(arr_tm)} 도착")
    elif network_routes:
        caption.append("\n🗓️ 신규 노선 상세:\n")
        for r in network_routes:
            eff_kr = format_date(r['eff_date'])
            s_net = get_korean_smart(r['start_en'], raw_text, 'airport')
            e_net = get_korean_smart(r['end_en'], raw_text, 'airport')
            freq_str = f"주 {r['count']}회" if r["unit"]=="weekly" else f"하루 {r['count']}회"
            line = f"  * {s_net} – {e_net}: {eff_kr}부터 {freq_str} 운항"
            if r['aircraft']: line += f" ({r['aircraft']})"
            caption.append(line)
    else:
        caption.append("\n🗓️ 상세 운항 스케줄은 항공사 홈페이지를 참고해주세요.")
        
    caption.append(f"\n🔗 출처: [AeroRoutes] {link}")
    caption.append("📸")
    return "\n".join(caption)

# ==========================================
# 3. UI 구성
# ==========================================
st.title("✈️ Aero-Post Generator")
st.markdown("항공 뉴스 캡션 자동 생성기 (Instagram Format)")

if not nlp:
    st.error("모델을 불러오지 못했습니다. data 폴더와 model 폴더를 확인해주세요.")
else:
    tab1, tab2 = st.tabs(["🔗 링크로 생성", "📝 텍스트로 생성"])

    with tab1:
        url_input = st.text_input("AeroRoutes 기사 URL")
        if st.button("캡션 생성 (Link)"):
            if url_input:
                with st.spinner('크롤링 중...'):
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        res = requests.get(url_input, headers=headers)
                        soup = BeautifulSoup(res.text, 'html.parser')
                        title_tag = soup.find('h1', class_='blog-title')
                        if not title_tag: title_tag = soup.find('h1', class_='entry-title')
                        title = title_tag.get_text(strip=True) if title_tag else "제목 없음"
                        
                        content = ""
                        for cls in ['entry-content', 'sqs-block-content', 'BlogList-item-excerpt']:
                            div = soup.find('div', class_=cls)
                            if div: content = div.get_text("\n", strip=True); break
                        
                        if content:
                            result = generate_caption(title, content, url_input)
                            st.success("생성 완료!")
                            st.text_area("결과 (복사해서 사용하세요)", value=result, height=400)
                        else:
                            st.error("본문을 찾을 수 없습니다.")
                    except Exception as e:
                        st.error(f"에러 발생: {e}")
    
    with tab2:
        title_in = st.text_input("제목 (선택사항)", value="항공 뉴스")
        text_in = st.text_area("기사 본문", height=200)
        if st.button("캡션 생성 (Text)"):
            if text_in:
                with st.spinner('분석 중...'):
                    result = generate_caption(title_in, text_in, "https://www.aeroroutes.com")
                    st.success("생성 완료!")
                    st.text_area("결과", value=result, height=400)