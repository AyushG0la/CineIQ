import streamlit as st
import pandas as pd
import requests
import json
import os
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── API Keys ──────────────────────────────────────────────
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
MOVIES_FILE = os.path.join(os.path.dirname(__file__), "movies.json")

# ── Base Movies ───────────────────────────────────────────
base_movies = [
    {"title": "Mujhse Shaadi Karogi", "genres": "bollywood hindi romance comedy slapstick fun salman akshay 2000s mistaken identity silly lighthearted", "poster": "🎭"},
    {"title": "Dhoom 2", "genres": "bollywood hindi action thriller heist bike chase hrithik aishwarya 2000s stylish slick stunt", "poster": "🏍️"},
    {"title": "Kal Ho Na Ho", "genres": "bollywood hindi romance drama emotional sad friendship shahrukh preity saif 2000s new york terminal illness", "poster": "💔"},
    {"title": "Guru", "genres": "bollywood hindi drama biography inspiring businessman mani ratnam abhishek aishwarya 2000s entrepreneurship ambition", "poster": "👔"},
    {"title": "3 Idiots", "genres": "bollywood hindi comedy drama inspiring college friendship aamir rajkumar hirani 2000s education system", "poster": "🎓"},
    {"title": "Saiyaara", "genres": "bollywood hindi romance drama emotional love 2020s young couple music longing", "poster": "🌙"},
    {"title": "Kabir Singh", "genres": "bollywood hindi romance drama intense emotional obsessive love shahid 2010s medical college toxic passion", "poster": "💊"},
    {"title": "Love Aaj Kal", "genres": "bollywood hindi romance modern love emotional 2010s parallel story kartik sara nostalgia", "poster": "💝"},
    {"title": "Mohabbatein", "genres": "bollywood hindi romance drama classic emotional love college shahrukh amitabh 2000s tradition rebellion forbidden", "poster": "🎻"},
    {"title": "Haunted", "genres": "bollywood hindi horror thriller scary supernatural ghost period 2010s haunted house possession eerie", "poster": "👻"},
    {"title": "Raaz", "genres": "bollywood hindi horror thriller scary supernatural mystery ghost bipasha 2000s haunted revenge dark secret past", "poster": "🔮"},
    {"title": "Real Steel", "genres": "hollywood english action drama inspiring robot boxing father son emotional underdog sports redemption jackman", "poster": "🤖"},
    {"title": "Titanic", "genres": "hollywood english romance drama emotional classic sad ship disaster leonardo dicaprio kate 1990s class love loss", "poster": "🚢"},
    {"title": "Zathura", "genres": "hollywood english adventure fantasy family fun kids space board game siblings 2000s jumanji", "poster": "🚀"},
    {"title": "Focus", "genres": "hollywood english thriller action heist con smart stylish will smith margot robbie 2010s deception manipulation", "poster": "🎯"},
    {"title": "Bird Box", "genres": "hollywood english horror thriller scary survival post apocalyptic sandra bullock 2010s blindfold creature mystery", "poster": "🐦"},
    {"title": "RRR", "genres": "south indian telugu action drama intense powerful historical rebellion british india friendship jr ntr ram charan 2020s", "poster": "🔥"},
    {"title": "Pathaan", "genres": "bollywood hindi action thriller intense spy adventure shahrukh deepika 2020s terrorism mission", "poster": "🕵️"},
    {"title": "Inception", "genres": "hollywood english thriller mystery smart intense dreamworld sci-fi christopher nolan dicaprio 2010s psychological layers heist", "poster": "🌀"},
    {"title": "Interstellar", "genres": "hollywood english drama adventure emotional intense sci-fi space time dilation christopher nolan 2010s fatherhood survival wormhole", "poster": "🌌"},
    {"title": "The Dark Knight", "genres": "hollywood english action thriller intense smart superhero batman joker christopher nolan 2000s crime gotham psychological anarchist", "poster": "🦇"},
    {"title": "Dil Chahta Hai", "genres": "bollywood hindi comedy drama friendship classic aamir saif akshaye 2000s goa coming of age urban lifestyle carefree", "poster": "🌊"},
    {"title": "Dangal", "genres": "bollywood hindi drama biography inspiring sports wrestling father daughters aamir 2010s haryana national champion girls empowerment", "poster": "🏆"},
    {"title": "Sultan", "genres": "bollywood hindi drama action inspiring sports wrestling comeback salman 2010s emotional redemption mma relationship", "poster": "🥊"},
    {"title": "PK", "genres": "bollywood hindi comedy drama inspiring fun emotional alien satire aamir anushka 2010s religion society question", "poster": "👽"},
    {"title": "Chennai Express", "genres": "bollywood hindi comedy romance fun adventure south india shahrukh deepika 2010s train goon chase runaway", "poster": "🚂"},
    {"title": "Andhadhun", "genres": "bollywood hindi thriller mystery smart suspense dark comedy ayushmann 2010s blind pianist murder neo noir unexpected twist", "poster": "🎹"},
    {"title": "Article 15", "genres": "bollywood hindi drama thriller intense powerful caste discrimination social justice ayushmann 2010s investigative police rural india", "poster": "📜"},
    {"title": "Gangs of Wasseypur", "genres": "bollywood hindi action drama intense powerful crime revenge family saga coal mafia anurag kashyap 2010s gritty raw", "poster": "🔫"},
    {"title": "Dil Dhadakne Do", "genres": "bollywood hindi drama comedy family emotional rich cruise ship priyanka ranveer 2010s dysfunctional wealthy relationships", "poster": "⛵"},
    {"title": "Zindagi Na Milegi Dobara", "genres": "bollywood hindi drama comedy friendship adventure spain europe hrithik farhan abhay 2010s road trip bucket list self discovery", "poster": "🏄"},
    {"title": "Queen", "genres": "bollywood hindi drama inspiring emotional journey solo travel self discovery kangana 2010s paris amsterdam independence girl", "poster": "👸"},
    {"title": "Stree", "genres": "bollywood hindi horror comedy fun scary supernatural folk legend shraddha rajkumar 2010s small town ghost humor", "poster": "🌿"},
    {"title": "Bhediya", "genres": "bollywood hindi horror comedy fun scary supernatural werewolf varun dhawan 2020s arunachal northeast creature humor", "poster": "🐺"},
    {"title": "The Avengers", "genres": "hollywood english action adventure superhero marvel ensemble iron man thor hulk 2010s team save world alien invasion", "poster": "🦸"},
    {"title": "Spider-Man No Way Home", "genres": "hollywood english action adventure superhero marvel emotional multiverse tom holland 2020s nostalgia multiversal villains", "poster": "🕷️"},
    {"title": "Top Gun Maverick", "genres": "hollywood english action drama intense inspiring adventure jet pilot tom cruise 2020s legacy friendship sacrifice flight", "poster": "✈️"},
    {"title": "The Pursuit of Happyness", "genres": "hollywood english drama biography inspiring emotional father son will smith 2000s poverty homelessness perseverance wall street stockbroker", "poster": "💼"},
    {"title": "A Beautiful Mind", "genres": "hollywood english drama biography inspiring emotional intense mathematics schizophrenia russell crowe 2000s princeton genius mental illness", "poster": "🧠"},
    {"title": "Forrest Gump", "genres": "hollywood english drama comedy emotional classic inspiring simple man history tom hanks 1990s journey love loss war running", "poster": "🍫"},
    {"title": "Karthik Calling Karthik", "genres": "bollywood hindi drama mystery romance psychological thriller farhan akhtar 2010s identity mental illness suspense dark", "poster": "🎭"},
    {"title": "Kabhi Khushi Kabhie Gham", "genres": "bollywood hindi romance drama family emotional shahrukh amitabh kareena 2000s grandeur wealth separation reunion tradition", "poster": "👨‍👩‍👧‍👦"},
    {"title": "Golmaal Fun Unlimited", "genres": "bollywood hindi comedy slapstick fun friendship ajay rohit shetty 2000s absurd scheme blind man trio chaos", "poster": "🎪"},
    {"title": "Dilwale Dulhania Le Jayenge", "genres": "bollywood hindi romance drama classic emotional love shahrukh kajol 1990s europe train family approval", "poster": "❤️"},
    {"title": "Partner", "genres": "bollywood hindi comedy romance fun buddy govinda salman 2000s lighthearted silly love advice matchmaking", "poster": "🤝"},
    {"title": "Pushpa The Rise", "genres": "south indian telugu action drama intense thriller smuggling allu arjun forest red sandalwood attitude rebellion mass 2020s", "poster": "🌺"},
    {"title": "Baahubali The Beginning", "genres": "south indian telugu action drama intense powerful historical epic fantasy kingdom war prabhas 2010s mythological scale", "poster": "👑"},
    {"title": "KGF Chapter 1", "genres": "south indian kannada action drama intense powerful gangster rocky 1970s gold mines revenge rise mass yash", "poster": "⛏️"},
    {"title": "KGF Chapter 2", "genres": "south indian kannada action drama intense powerful gangster yash sequel revenge bloodbath empire mass 2020s", "poster": "⛏️"},
    {"title": "Phir Hera Pheri", "genres": "bollywood hindi comedy slapstick fun trio akshay suniel paresh 2000s sequel money scheme absurd debt chaos", "poster": "😂"},
    {"title": "Dead Silence", "genres": "hollywood english horror mystery thriller supernatural ventriloquist doll curse 2000s eerie silent revenge ghost town", "poster": "🎭"},
    {"title": "Mama", "genres": "hollywood english horror fantasy thriller supernatural creature ghost children 2010s orphan dark eerie haunting jungle", "poster": "🎭"},
    {"title": "The Conjuring", "genres": "hollywood english horror mystery thriller supernatural true story demonic haunted house 2010s warrens paranormal investigation", "poster": "🎭"},
    {"title": "The Conjuring 2", "genres": "hollywood english horror mystery thriller supernatural true story demonic haunted house 2010s warrens enfield paranormal sequel", "poster": "🎭"},
    {"title": "The Conjuring The Devil Made Me Do It", "genres": "hollywood english horror mystery thriller supernatural true story courtroom demonic possession 2020s warrens murder", "poster": "🎭"},
    {"title": "The Conjuring Last Rites", "genres": "hollywood english horror mystery thriller supernatural demonic warrens 2020s paranormal investigation curse ritual", "poster": "🎭"},
    {"title": "Final Destination", "genres": "hollywood english horror thriller death premonition teenagers 2000s gruesome accidents fate survival escape", "poster": "🎭"},
    {"title": "Final Destination 2", "genres": "hollywood english horror thriller death premonition highway accident 2000s gruesome fate survival sequel", "poster": "🎭"},
    {"title": "Final Destination 3", "genres": "hollywood english horror thriller death premonition roller coaster 2000s gruesome fate survival sequel", "poster": "🎭"},
    {"title": "Final Destination 5", "genres": "hollywood english horror thriller death premonition bridge 2010s gruesome fate survival sequel bridge collapse", "poster": "🎭"},
    {"title": "Chak De India", "genres": "bollywood hindi drama sports inspiring team hockey women national shahrukh 2000s comeback redemption coach underdog", "poster": "🎭"},
    {"title": "Chhichhore", "genres": "bollywood hindi comedy drama friendship college nostalgia sushant 2010s parallel past present engineering hostel life", "poster": "🎭"},
    {"title": "Secret Superstar", "genres": "bollywood hindi drama music inspiring young girl dreams aamir 2010s singing youtube family struggle mother daughter", "poster": "🎭"},
    {"title": "Hichki", "genres": "bollywood hindi comedy drama inspiring teacher school rani mukerji 2010s tourette syndrome classroom underdog students", "poster": "🎭"},
    {"title": "12th Fail", "genres": "bollywood hindi biography drama inspiring upsc exam honest officer vikrant massey 2020s struggle small town poverty perseverance", "poster": "🎭"},
    {"title": "Bhaag Milkha Bhaag", "genres": "bollywood hindi biography drama sports inspiring track running farhan akhtar 2010s partition trauma national athlete flying sikh", "poster": "🎭"},
    {"title": "Super 30", "genres": "bollywood hindi biography drama inspiring education poor students iit coaching hrithik 2010s mathematician bihar struggle", "poster": "🎭"},
    {"title": "83", "genres": "bollywood hindi biography drama sports inspiring cricket world cup 1983 kapil dev ranveer 2020s underdog historic victory team", "poster": "🎭"},
    {"title": "Lakshya", "genres": "bollywood hindi action drama romance inspiring army military kargil hrithik preity 2000s young man purpose war growth", "poster": "🎭"},
    {"title": "Chandu Champion", "genres": "bollywood hindi action biography drama inspiring sports special olympics disability kartik 2020s true story athlete", "poster": "🎭"},
    {"title": "Soorma", "genres": "bollywood hindi biography drama sports inspiring hockey diljit 2010s shooting accident comeback sandeep singh real story", "poster": "🎭"},
    {"title": "Swades", "genres": "bollywood hindi drama inspiring emotional patriotism nasa scientist rural village shahrukh 2000s return roots development", "poster": "🎭"},
    {"title": "Rang De Basanti", "genres": "bollywood hindi drama comedy historical patriotism youth friendship aamir 2000s revolution parallel past present sacrifice", "poster": "🎭"},
    {"title": "Lagaan", "genres": "bollywood hindi drama sports musical historical cricket british colonial aamir 2000s village tax challenge underdog", "poster": "🎭"},
    {"title": "Pardes", "genres": "bollywood hindi drama romance musical shahrukh 1990s india america culture clash family values NRI tradition", "poster": "🎭"},
    {"title": "Raja Hindustani", "genres": "bollywood hindi comedy drama romance fun aamir 1990s taxi driver rich girl love class difference simple", "poster": "🎭"},
    {"title": "Dilwale", "genres": "bollywood hindi action comedy drama romance shahrukh kajol rohit shetty 2010s past love reunion gangster family", "poster": "🎭"},
    {"title": "Dulhe Raja", "genres": "bollywood hindi comedy slapstick fun govinda 1990s dhaba canteen rivalry absurd lighthearted silly", "poster": "🎭"},
    {"title": "Coolie No 1", "genres": "bollywood hindi comedy romance fun govinda 1990s class difference arranged marriage slapstick silly lighthearted mistaken identity", "poster": "🎭"},
    {"title": "Hero No 1", "genres": "bollywood hindi comedy musical fun govinda 1990s cross dressing disguise silly slapstick mistaken identity lighthearted", "poster": "🎭"},
    {"title": "Biwi No 1", "genres": "bollywood hindi comedy drama romance fun salman sushmita 1990s extramarital affair jealousy slapstick family silly", "poster": "🎭"},
    {"title": "Kuch Kuch Hota Hai", "genres": "bollywood hindi comedy drama romance musical shahrukh kajol rani 1990s college friendship love triangle reunion classic", "poster": "🎭"},
    {"title": "Dil To Pagal Hai", "genres": "bollywood hindi comedy drama romance musical shahrukh madhuri karisma 1990s dance love triangle friendship emotional", "poster": "🎭"},
    {"title": "Sam Bahadur", "genres": "bollywood hindi biography drama war military inspiring vicky kaushal 2020s sam manekshaw army chief india pakistan 1971", "poster": "🎭"},
    {"title": "Shershaah", "genres": "bollywood hindi action biography drama war military inspiring sidharth malhotra 2020s captain vikram batra kargil love story sacrifice", "poster": "🎭"},
    {"title": "Gunjan Saxena The Kargil Girl", "genres": "bollywood hindi biography drama inspiring female pilot janhvi 2020s air force kargil war gender barrier aviation", "poster": "🎭"},
    {"title": "Uri The Surgical Strike", "genres": "bollywood hindi action drama history military inspiring vicky kaushal 2010s surgical strike pakistan army patriotism", "poster": "🎭"},
    {"title": "Article 370", "genres": "bollywood hindi action drama thriller political yami gautam 2020s kashmir intelligence mission government decision", "poster": "🎭"},
    {"title": "Neerja", "genres": "bollywood hindi biography drama thriller inspiring flight attendant sonam kapoor 2010s hijack bravery real story pan am", "poster": "🎭"},
    {"title": "Raazi", "genres": "bollywood hindi action drama thriller spy inspiring alia bhatt 2010s pakistan india undercover agent 1971 war sacrifice", "poster": "🎭"},
    {"title": "Rocky Aur Rani Kii Prem Kahaani", "genres": "bollywood hindi comedy drama romance family ranveer alia karan johar 2020s cultural clash joint family love story grandparents", "poster": "🎭"},
    {"title": "Humpty Sharma Ki Dulhania", "genres": "bollywood hindi comedy romance drama fun varun alia 2010s college love story arranged marriage chase lighthearted", "poster": "🎭"},
    {"title": "Tu Jhoothi Main Makkaar", "genres": "bollywood hindi comedy romance fun modern ranbir shraddha 2020s breakup artist relationship games lighthearted urban", "poster": "🎭"},
    {"title": "Sonu Ke Titu Ki Sweety", "genres": "bollywood hindi comedy romance fun friendship bromance kartik 2010s best friends love rivalry bro code lighthearted", "poster": "🎭"},
    {"title": "Kapoor And Sons", "genres": "bollywood hindi drama family emotional fawad sidharth alia 2010s dysfunctional family secrets reunion death reconciliation", "poster": "🎭"},
    {"title": "Shiddat", "genres": "bollywood hindi romance drama emotional love obsessive journey sunny kaushal 2020s long distance separation longing", "poster": "🎭"},
    {"title": "Yeh Jawaani Hai Deewani", "genres": "bollywood hindi comedy drama romance musical friendship ranbir deepika 2010s travel adventure ambition love reunion nostalgia", "poster": "🎭"},
    {"title": "Dil Bechara", "genres": "bollywood hindi comedy drama romance emotional sushant 2020s cancer terminal illness love youth last days bittersweet", "poster": "🎭"},
    {"title": "Fukrey", "genres": "bollywood hindi comedy drama fun friendship scheme pulkit varun ali 2010s delhi college jugaad lottery funny chaos", "poster": "🎭"},
    {"title": "Hera Pheri", "genres": "bollywood hindi comedy slapstick fun trio akshay suniel paresh 2000s mistaken identity debt absurd scheme classic cult", "poster": "😂"},
    {"title": "Dhurandhar", "genres": "bollywood hindi action adventure comedy heist witty stylish ranveer singh 2020s con game chase smart fun thriller ensemble", "poster": "🎯"},
]

# ── File Storage ──────────────────────────────────────────
def load_movies():
    if os.path.exists(MOVIES_FILE):
        with open(MOVIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    save_movies(base_movies)
    return base_movies.copy()

def save_movies(movies):
    with open(MOVIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)

# ── TMDB API ──────────────────────────────────────────────
TITLE_FIXES = {
    "dhoom 2": "Dhoom 2: Back in Action",
    "dil bechara": "Dil Bechara",
    "kal ho na ho": "Kal Ho Naa Ho",
    "baahubali the beginning": "Baahubali: The Beginning",
    "baahubali the conclusion": "Baahubali: The Conclusion",
    "pushpa the rise": "Pushpa: The Rise - Part 1",
    "kgf chapter 1": "K.G.F: Chapter 1",
    "kgf chapter 2": "K.G.F: Chapter 2",
    "gunjan saxena the kargil girl": "Gunjan Saxena: The Kargil Girl",
    "uri the surgical strike": "URI: The Surgical Strike",
    "rocky aur rani kii prem kahaani": "Rocky Aur Rani Kii Prem Kahaani",
    "the conjuring the devil made me do it": "The Conjuring: The Devil Made Me Do It",
    "ek tha tiger": "Ek Tha Tiger",
    "tiger zinda hai": "Tiger Zinda Hai",
    "tiger 3": "Tiger 3",
    "don": "Don",
    "don 2": "Don 2",
    "agneepath": "Agneepath",
    "singham": "Singham",
    "rowdy rathore": "Rowdy Rathore",
    "dabangg": "Dabangg",
    "dabangg 2": "Dabangg 2",
    "dabangg 3": "Dabangg 3",
    "bajrangi bhaijaan": "Bajrangi Bhaijaan",
    "pk": "PK",
    "ms dhoni the untold story": "M.S. Dhoni: The Untold Story",
    "dunki": "Dunki",
    "jawan": "Jawan",
    "animal": "Animal",
    "kalki 2898 ad": "Kalki 2898 - AD",
    "stree 2": "Stree 2",
    "brahmastra": "Brahmastra: Part One \u2013 Shiva",
    "brahmastra part one shiva": "Brahmastra: Part One \u2013 Shiva",
    "vikram vedha": "Vikram Vedha",
    "gangubai kathiawadi": "Gangubai Kathiawadi",
    "atrangi re": "Atrangi Re",
    "sooryavanshi": "Sooryavanshi",
    "radhe": "Radhe: Your Most Wanted Bhai",
    "antim": "Antim: The Final Truth",
    "jersey": "Jersey",
    "prithviraj": "Samrat Prithviraj",
    "laal singh chaddha": "Laal Singh Chaddha",
    "shamshera": "Shamshera",
    "doctor strange multiverse of madness": "Doctor Strange in the Multiverse of Madness",
    "spiderman no way home": "Spider-Man: No Way Home",
    "spider man no way home": "Spider-Man: No Way Home",
    "avengers endgame": "Avengers: Endgame",
    "avengers infinity war": "Avengers: Infinity War",
}

def _raw_tmdb_search(query, year=None):
    all_results = []
    try:
        params = {"api_key": TMDB_API_KEY, "query": query, "include_adult": False}
        if year:
            params["year"] = year
        r = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=6, verify=False).json()
        all_results = r.get('results', [])
    except requests.exceptions.RequestException:
        pass
    return all_results

def _title_variants(title):
    t = title.strip()
    variants = []
    fix = TITLE_FIXES.get(t.lower())
    if fix:
        variants.append(fix)
    variants.append(t)
    no_punct = re.sub(r'[^\w\s]', '', t).strip()
    if no_punct and no_punct not in variants:
        variants.append(no_punct)
    stripped = re.sub(r'\s+\d+$', '', t).strip()
    if stripped and stripped != t and stripped not in variants:
        variants.append(stripped)
    words = t.split()
    if len(words) > 3:
        short = ' '.join(words[:3])
        if short not in variants:
            variants.append(short)
    return variants

# ── VERIFIED TMDB IDs ─────────────────────────────────────
# FIX 1: Added missing Bollywood movies that were falling through to wrong variant search
TMDB_IDS = {
    # Tiger / YRF Spy Universe
    "ek tha tiger": 85985,
    "tiger zinda hai": 441909,
    "tiger 3": 933260,
    # SRK films
    "dilwale": 370665,
    "don": 17501,
    "don 2": 62213,
    # Salman films
    "dabangg": 44425,
    "dabangg 2": 147405,
    "dabangg 3": 601226,
    "bajrangi bhaijaan": 348892,
    "radhe": 776503,
    "biwi no 1": 27194,
    "partner": 13477,
    "mujhse shaadi karogi": 21597,
    "sultan": 332979,
    # Aamir films
    "pk": 297222,
    "laal singh chaddha": 601635,
    "lagaan": 20836,
    "rang de basanti": 14549,
    "swades": 27215,
    "3 idiots": 20453,
    "dangal": 360814,
    "secret superstar": 463272,
    # Recent blockbusters
    "jawan": 872906,
    "animal": 781732,
    "dunki": 960876,
    "stree 2": 1112426,
    "dhurandhar": 1291608,
    "brahmastra": 675353,
    "brahmastra part one shiva": 675353,
    "sooryavanshi": 592508,
    "gangubai kathiawadi": 664332,
    "kalki 2898 ad": 1001562,
    "pathaan": 849260,
    "stree": 534979,
    "bhediya": 891699,
    "fukrey": 236311,
    # Biopics / War films
    "ms dhoni the untold story": 388333,
    "sanju": 496328,
    "uri the surgical strike": 587664,
    "sam bahadur": 1086747,
    "article 370": 1087822,
    "shershaah": 780426,
    "gunjan saxena the kargil girl": 694978,
    "neerja": 371741,
    "raazi": 499171,
    "bhaag milkha bhaag": 163280,
    "soorma": 492044,
    "83": 588977,
    "super 30": 553182,
    "chandu champion": 1094512,
    "12th fail": 1167268,
    "lakshya": 27207,
    # Common name-clash films
    "agneepath": 70283,
    "singham": 68728,
    "rowdy rathore": 89722,
    "jersey": 765360,
    "fighter": 1045938,
    "bade miyan chote miyan": 1209290,
    # 90s / 2000s classics
    "dilwale dulhania le jayenge": 19104,
    "kuch kuch hota hai": 19602,
    "kabhi khushi kabhie gham": 24886,
    "mohabbatein": 11518,
    "kal ho na ho": 24884,
    "dil chahta hai": 29101,
    "dil to pagal hai": 21626,
    "pardes": 27193,
    "raja hindustani": 27197,
    "dulhe raja": 27190,
    "coolie no 1": 27192,          # 1995 Govinda film
    "hero no 1": 27198,
    "golmaal fun unlimited": 13469,
    "hera pheri": 28302,
    "phir hera pheri": 44751,
    "dhoom 2": 14564,
    # FIX: "Haunted" was missing → was returning a Hollywood film
    "haunted": 80230,              # Haunted 3D (2011) - Bollywood
    # FIX: "Guru" was missing → ambiguous title risk
    "guru": 9603,                  # Guru (2007) - Mani Ratnam / Abhishek Bachchan
    # FIX: "Raaz" was missing
    "raaz": 29156,                 # Raaz (2002) - Bipasha Basu
    # 2010s-2020s Bollywood
    "kabir singh": 598045,
    "love aaj kal": 508965,
    "andhadhun": 573436,
    "article 15": 622122,
    "gangs of wasseypur": 96724,
    "dil dhadakne do": 301681,
    "zindagi na milegi dobara": 80304,
    "queen": 248981,
    "yeh jawaani hai deewani": 152601,
    "dil bechara": 680833,
    "chhichhore": 630590,
    "hichki": 476292,
    "karthik calling karthik": 47426,
    "kapoor and sons": 381008,
    "sonu ke titu ki sweety": 496894,
    "humpty sharma ki dulhania": 244201,
    "tu jhoothi main makkaar": 976891,
    "rocky aur rani kii prem kahaani": 882598,
    "shiddat": 713149,
    "chennai express": 157820,
    "atrangi re": 826726,
}

# FIX 2: Changed ttl=0 → ttl=3600 (1-hour cache)
# ttl=0 means INFINITE cache in Streamlit — stale wrong posters never refresh!
@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_search(title):
    """
    Smart TMDB search:
    1. Direct TMDB ID lookup for known titles (bypasses title mismatch)
    2. TITLE_FIXES + auto-generated title variants fallback
    """
    t = title.strip().lower()

    # 1. Direct ID lookup
    if t in TMDB_IDS:
        tmdb_id = TMDB_IDS[t]
        if tmdb_id:  # FIX 3: Guard against None IDs that would crash the request
            try:
                r = requests.get(
                    f"{TMDB_BASE}/movie/{tmdb_id}",
                    params={"api_key": TMDB_API_KEY},
                    timeout=6, verify=False).json()
                if r.get('id'):
                    return r
            except requests.exceptions.RequestException:
                pass  # Fall through to variant search

    # 2. Variant-based search with poster preference
    best_with_poster = None
    best_without_poster = None

    for variant in _title_variants(title):
        for result in _raw_tmdb_search(variant):
            if result.get('poster_path'):
                if best_with_poster is None:
                    best_with_poster = result
                    break
            elif best_without_poster is None:
                best_without_poster = result
        if best_with_poster:
            break

    return best_with_poster or best_without_poster or None

@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_details(tmdb_id):
    """Full movie details: genres + keywords + credits in one call."""
    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY, "append_to_response": "keywords,credits"},
            timeout=6, verify=False).json()
        return r
    except requests.exceptions.RequestException:
        return {}

def build_rich_genres_tmdb(details):
    parts = []
    for g in details.get('genres', []):
        parts.append(g['name'].lower())
    for k in details.get('keywords', {}).get('keywords', [])[:15]:
        parts.append(k['name'].lower().replace('-', ' '))
    for actor in details.get('credits', {}).get('cast', [])[:3]:
        parts.append(actor['name'].lower())
    crew = details.get('credits', {}).get('crew', [])
    director = next((c['name'].lower() for c in crew if c['job'] == 'Director'), '')
    if director:
        parts.append(director)
    lang = details.get('original_language', '')
    origin_names = [c['name'] for c in details.get('production_countries', [])]
    if lang == 'hi':
        parts.append('bollywood hindi')
    elif lang in ['te', 'ta', 'kn', 'ml']:
        lang_map = {'te': 'telugu', 'ta': 'tamil', 'kn': 'kannada', 'ml': 'malayalam'}
        parts.append(f"south indian {lang_map[lang]}")
    elif lang == 'en':
        parts.append('hollywood english')
    elif 'India' in origin_names:
        parts.append('indian')
    year = details.get('release_date', '')[:4]
    if year and year.isdigit():
        parts.append(year[:3] + '0s')
    return ' '.join(parts).strip()

@st.cache_data(show_spinner=False, ttl=3600)
def _get_trailer_url(tmdb_id, title_fallback):
    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{tmdb_id}/videos",
            params={"api_key": TMDB_API_KEY}, timeout=6, verify=False).json()
        videos = r.get('results', [])
        for v in videos:
            if v.get('site') == 'YouTube' and v.get('type') == 'Trailer' and v.get('official'):
                return f"https://www.youtube.com/watch?v={v['key']}"
        for v in videos:
            if v.get('site') == 'YouTube' and v.get('type') == 'Trailer':
                return f"https://www.youtube.com/watch?v={v['key']}"
        for v in videos:
            if v.get('site') == 'YouTube':
                return f"https://www.youtube.com/watch?v={v['key']}"
    except requests.exceptions.RequestException:
        pass
    return f"https://www.youtube.com/results?search_query={title_fallback.replace(' ', '+')}+official+trailer"

@st.cache_data(show_spinner=False, ttl=3600)
def get_movie_data(title):
    """Full movie data for display + rich_genres for saving to DB."""
    try:
        result = tmdb_search(title)
        if not result:
            return {'found': False, 'poster': None, 'rich_genres': ''}
        tmdb_id = result['id']
        details = tmdb_details(tmdb_id)
        poster_path = details.get('poster_path') or result.get('poster_path')
        poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None
        rich_genres = build_rich_genres_tmdb(details)
        cast_list = details.get('credits', {}).get('cast', [])
        cast_str = ', '.join(c['name'] for c in cast_list[:6])
        crew = details.get('credits', {}).get('crew', [])
        director = next((c['name'] for c in crew if c['job'] == 'Director'), 'N/A')
        kw_list = details.get('keywords', {}).get('keywords', [])
        keywords_str = ', '.join(k['name'] for k in kw_list[:8])
        rating = details.get('vote_average', 'N/A')
        if rating != 'N/A':
            rating = round(float(rating), 1)
        return {
            'found': True,
            'poster': poster_url,
            'rich_genres': rich_genres,
            'rating': rating,
            'plot': details.get('overview', 'N/A'),
            'cast': cast_str or 'N/A',
            'year': details.get('release_date', '')[:4] or 'N/A',
            'director': director,
            'keywords': keywords_str,
            'tmdb_id': tmdb_id,
            'tmdb_url': f"https://www.themoviedb.org/movie/{tmdb_id}",
            'trailer_url': _get_trailer_url(tmdb_id, title)
        }
    except (requests.exceptions.RequestException, KeyError, TypeError, ValueError):
        return {'found': False, 'poster': None, 'rich_genres': ''}

# ── Similarity Engine ─────────────────────────────────────
def get_industry(g):
    g = g.lower()
    if any(k in g for k in ['bollywood', 'hindi']): return 'bollywood'
    if any(k in g for k in ['south indian', 'telugu', 'tamil', 'kannada', 'malayalam']): return 'south'
    if any(k in g for k in ['hollywood', 'english']): return 'hollywood'
    return 'other'

def get_genre_set(g):
    tags = ['action', 'comedy', 'romance', 'horror', 'thriller', 'drama',
            'biography', 'adventure', 'fantasy', 'sport', 'mystery', 'crime',
            'musical', 'war', 'history', 'science fiction', 'animation']
    return set(t for t in tags if t in g.lower())

# FIX 4: Removed @st.cache_data from build_similarity — the tuple hash was
# sometimes stale when movies were added mid-session, causing old recommendations.
# It now always recomputes fresh; the cost is trivial for <200 movies.
def build_similarity(movies_tuple):
    movies = [dict(m) for m in movies_tuple]
    df = pd.DataFrame(movies)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True)
    matrix = tfidf.fit_transform(df['genres'])
    cosine_sim = cosine_similarity(matrix)
    n = len(df)
    adjustment = np.zeros((n, n))
    industries = [get_industry(g) for g in df['genres']]
    genre_sets = [get_genre_set(g) for g in df['genres']]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if industries[i] != industries[j]:
                adjustment[i][j] -= 0.25
            gi, gj = genre_sets[i], genre_sets[j]
            if gi and gj:
                overlap = len(gi & gj)
                if overlap == 0:
                    adjustment[i][j] -= 0.15
                elif overlap >= 2:
                    adjustment[i][j] += 0.08
    return df, np.clip(cosine_sim + adjustment, 0, 1)

def recommend(title, movies, n=5):
    movies_tuple = tuple(tuple(sorted(m.items())) for m in movies)
    df, sim = build_similarity(movies_tuple)
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        return []
    idx = matches.index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    candidate_scores = [s for _, s in scores[1:] if s > 0]
    if not candidate_scores:
        return []
    top_score = candidate_scores[0]
    dynamic_min = max(top_score * 0.65, 0.15)
    results = []
    for i, score in scores[1:]:
        if score < dynamic_min:
            continue
        results.append((df['title'][i], round(score * 100)))
        if len(results) >= n:
            break
    return results

# ── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="🎬 CineIQ", page_icon="🎬", layout="wide")

# ── Session State ─────────────────────────────────────────
if 'movies' not in st.session_state: st.session_state.movies = load_movies()
if 'search_result' not in st.session_state: st.session_state.search_result = None
if 'search_query_saved' not in st.session_state: st.session_state.search_query_saved = ""
if 'dev_logged_in' not in st.session_state: st.session_state.dev_logged_in = False
if 'login_attempts' not in st.session_state: st.session_state.login_attempts = 0

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
st.sidebar.divider()
st.sidebar.metric("Total Movies", len(st.session_state.movies))
st.sidebar.divider()
st.sidebar.subheader("🔐 Developer Zone")

if not st.session_state.dev_logged_in:
    dev_password = st.sidebar.text_input("Developer Password:", type="password", placeholder="Enter password...")
    if st.sidebar.button("🔓 Login", use_container_width=True):
        if st.session_state.login_attempts >= 5:
            st.sidebar.error("🔒 Too many attempts. Refresh to try again.")
        elif dev_password == ADMIN_PASSWORD:
            st.session_state.dev_logged_in = True
            st.session_state.login_attempts = 0
            st.rerun()
        else:
            st.session_state.login_attempts += 1
            remaining = 5 - st.session_state.login_attempts
            st.sidebar.error(f"❌ Wrong password! {remaining} attempt(s) left.")
else:
    st.sidebar.success("✅ Developer logged in!")

    # FIX 5: Added "Clear Poster Cache" button in admin panel
    # This lets you force-refresh stale cached posters without redeploying
    if st.sidebar.button("🔄 Clear Poster Cache", use_container_width=True, help="Force re-fetch all posters from TMDB"):
        tmdb_search.clear()
        tmdb_details.clear()
        get_movie_data.clear()
        _get_trailer_url.clear()
        st.sidebar.success("✅ Cache cleared! Posters will refresh.")
        st.rerun()

    st.sidebar.subheader("🗑️ Remove Movie")
    movie_titles = [m['title'] for m in st.session_state.movies]
    movie_to_remove = st.sidebar.selectbox("Select movie:", ["-- Select --"] + movie_titles)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.sidebar.button("🗑️ Remove", use_container_width=True):
            if movie_to_remove == "-- Select --":
                st.sidebar.error("Select a movie!")
            else:
                st.session_state.movies = [m for m in st.session_state.movies if m['title'] != movie_to_remove]
                save_movies(st.session_state.movies)
                st.sidebar.success("✅ Removed!")
                st.rerun()
    with c2:
        if st.sidebar.button("🔒 Logout", use_container_width=True):
            st.session_state.dev_logged_in = False
            st.rerun()

# ── Dark Mode ─────────────────────────────────────────────
if dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; color: #fafafa !important; }
    section[data-testid="stSidebar"] { background-color: #262730 !important; }
    section[data-testid="stSidebar"] * { color: #fafafa !important; }
    label, p, span, div { color: #fafafa !important; }
    .stCaption, .stCaption * { color: #aaaaaa !important; }
    div[data-testid="stMetricValue"] { color: #00d4ff !important; }
    div[data-testid="stMetricLabel"] { color: #aaaaaa !important; }
    hr { border-color: #444 !important; }
    .stButton > button { background-color: #1e2130 !important; color: #fafafa !important; border: 1px solid #555 !important; }
    .stButton > button p, .stButton > button span { color: #fafafa !important; }
    div[data-baseweb="select"] { background-color: #1e2130 !important; }
    div[data-baseweb="select"] * { background-color: #1e2130 !important; color: #fafafa !important; }
    div[data-baseweb="select"] input { color: #fafafa !important; }
    div[data-baseweb="select"] svg { fill: #fafafa !important; }
    div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #1e2130 !important; }
    div[data-baseweb="popover"] *, div[data-baseweb="menu"] * { background-color: #1e2130 !important; color: #fafafa !important; }
    li[role="option"]:hover { background-color: #2e3250 !important; }
    .stTextInput input { background-color: #1e2130 !important; color: #fafafa !important; border-color: #444 !important; }
    .stTextInput input::placeholder { color: #888 !important; }
    .stLinkButton a { background-color: #1e2130 !important; color: #fafafa !important; border: 1px solid #555 !important; }
    .stLinkButton a:hover { background-color: #2e3250 !important; color: #ffffff !important; }
    .stLinkButton a p, .stLinkButton a span { color: #fafafa !important; }
    </style>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.title("🎬 CineIQ")
st.caption("Smart Movie Recommendations · Powered by TMDB + ML")
st.divider()

# ── Search ────────────────────────────────────────────────
st.subheader("🔍 Add a Movie")
col1, col2 = st.columns([4, 1])
with col1:
    search_query = st.text_input("Search:", placeholder="Type movie name e.g. Inception, Jawan...", label_visibility="collapsed")
with col2:
    search_btn = st.button("🔍 Search", use_container_width=True)

if search_btn and search_query:
    existing = [m['title'].lower() for m in st.session_state.movies]
    if search_query.lower() in existing:
        st.success(f"✅ '{search_query}' is already in database!")
        st.session_state.search_result = None
    else:
        with st.spinner("Fetching from TMDB..."):
            movie_data = get_movie_data(search_query)
        st.session_state.search_result = movie_data
        st.session_state.search_query_saved = search_query

# ── Show Search Result ────────────────────────────────────
if st.session_state.search_result:
    movie_data = st.session_state.search_result
    query = st.session_state.search_query_saved

    if movie_data['found']:
        col1, col2 = st.columns([1, 3])
        with col1:
            if movie_data['poster']:
                st.image(movie_data['poster'], width=150)
            else:
                st.markdown("## 🎬")
        with col2:
            st.success(f"✅ Found '{query}' on TMDB!")
            st.caption(f"📅 {movie_data.get('year','N/A')} | ⭐ {movie_data.get('rating','N/A')}/10 | 🎬 {movie_data.get('director','N/A')}")
            st.caption(f"👥 {movie_data.get('cast','N/A')}")
            st.write("**Auto-built tags (saved to DB):**")
            st.code(movie_data['rich_genres'], language=None)
            if movie_data.get('keywords'):
                st.caption(f"🏷️ TMDB Keywords: {movie_data['keywords']}")
            st.info("💡 **Optional:** Add tone/vibe words — e.g. `dark gritty emotional cult slapstick`")
            extra_tags = st.text_input("Extra tags (optional):", placeholder="e.g. dark gritty emotional")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Add to CineIQ", use_container_width=True):
                    full_genres = movie_data['rich_genres']
                    if extra_tags.strip():
                        full_genres += " " + extra_tags.strip()
                    st.session_state.movies.append({"title": query, "genres": full_genres, "poster": "🎬"})
                    save_movies(st.session_state.movies)
                    st.session_state.search_result = None
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.search_result = None
                    st.rerun()
    else:
        st.warning(f"⚠️ '{query}' not found on TMDB!")
        st.write("Add manually with rich tags:")
        st.info("💡 Include: language (`bollywood`/`hollywood`/`south indian`), genres, cast, era, tone")
        manual_genres = st.text_input("Tags:", placeholder="bollywood hindi action comedy heist ranveer 2020s fun witty")
        manual_poster = st.text_input("Emoji:", value="🎬")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ Add Manually", use_container_width=True):
                if manual_genres:
                    st.session_state.movies.append({"title": query, "genres": manual_genres, "poster": manual_poster})
                    save_movies(st.session_state.movies)
                    st.session_state.search_result = None
                    st.rerun()
                else:
                    st.error("Please enter tags!")
        with col_no:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.search_result = None
                st.rerun()

st.divider()

# ── Genre Filter ──────────────────────────────────────────
st.subheader("🎭 Filter by Genre")
genre_options = ["All", "romance", "action", "comedy", "thriller", "drama",
                 "horror", "biography", "bollywood", "hollywood", "south indian", "sport", "war"]
selected_genre = st.selectbox("Genre:", genre_options)

df_current = pd.DataFrame(st.session_state.movies)
filtered_df = df_current if selected_genre == "All" else \
    df_current[df_current['genres'].str.contains(selected_genre, case=False, na=False)]
st.caption(f"Showing {len(filtered_df)} movies")
st.divider()

# ── Movie Selection ───────────────────────────────────────
st.subheader("🎬 Select a Movie You Like")
col1, col2 = st.columns([3, 1])
with col1:
    movie_list = ["🎬 Select a movie..."] + filtered_df['title'].tolist()
    selected_raw = st.selectbox("Choose:", movie_list, label_visibility="collapsed")
    selected = None if selected_raw == "🎬 Select a movie..." else selected_raw
with col2:
    rec_choice = st.radio("Show:", ["Top 3", "Top 5"], horizontal=True)
    num_recs = 3 if rec_choice == "Top 3" else 5

if st.button("🎯 Get Recommendations", use_container_width=True):
    if not selected:
        st.warning("⚠️ Please select a movie first!")
    else:
        with st.spinner("CineIQ is thinking..."):
            recommendations = recommend(selected, st.session_state.movies, num_recs)
        st.divider()
        st.subheader(f"Because you liked **{selected}**:")
        st.write("")

        if not recommendations:
            st.warning("⚠️ Not enough similar movies found. Add more movies to improve recommendations!")
        else:
            all_movie_data = {t: get_movie_data(t) for t, s in recommendations}
            cols = st.columns(len(recommendations))
            for i, (title, score) in enumerate(recommendations):
                with cols[i]:
                    movie_data = all_movie_data[title]

                    # ── Poster ──
                    if movie_data.get('poster'):
                        st.image(movie_data['poster'], width=200)
                    else:
                        emoji = next((m.get('poster', '🎬') for m in st.session_state.movies if m['title'] == title), '🎬')
                        st.markdown(f"<h1 style='text-align:center'>{emoji}</h1>", unsafe_allow_html=True)

                    # ── Title + Score ──
                    st.markdown(f"**{title}**")
                    st.progress(score / 100)
                    if score >= 60:
                        st.markdown(f"🟢 **{score}% match**")
                    elif score >= 35:
                        st.markdown(f"🟡 **{score}% match**")
                    else:
                        st.markdown(f"🟠 **{score}% match**")

                    # ── Movie Details ──
                    if movie_data.get('found'):
                        if movie_data.get('rating') not in ('N/A', None):
                            st.markdown(f"⭐ **{movie_data['rating']}/10**")
                        if movie_data.get('year') not in ('N/A', None, ''):
                            st.caption(f"📅 {movie_data['year']}")
                        if movie_data.get('director') not in ('N/A', None, ''):
                            st.caption(f"🎬 {movie_data['director']}")
                        if movie_data.get('cast') not in ('N/A', None, ''):
                            cast_names = movie_data['cast'].split(', ')
                            cast_display = ', '.join(cast_names[:3])
                            st.caption(f"👥 {cast_display}")
                        if movie_data.get('plot') not in ('N/A', None, ''):
                            with st.expander("📝 Plot"):
                                st.write(movie_data['plot'])

                    # ── Buttons ──
                    col_t, col_i = st.columns(2)
                    with col_t:
                        if movie_data.get('trailer_url'):
                            st.link_button("▶️ Trailer", movie_data['trailer_url'], use_container_width=True)
                    with col_i:
                        if movie_data.get('tmdb_url'):
                            st.link_button("🎬 TMDB", movie_data['tmdb_url'], use_container_width=True)
