import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urlparse, urljoin
import time
import random
import concurrent.futures
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def fetch_page(url, retry_count=0):
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        return {"html": response.text, "final_url": response.url}
    except requests.exceptions.RequestException as e:
        status_code = getattr(e.response, 'status_code', None)
        if (status_code == 403 or status_code == 429) and retry_count < 3:
            logger.warning(f"Got {status_code} from {url}, retrying with different user agent...")
            time.sleep(1 + retry_count)
            return fetch_page(url, retry_count + 1)
        
        logger.error(f"Failed to fetch {url}: {e}")
        return None

def is_executive_title(title):
    if not title:
        return False
    lower_title = title.lower().strip()
    
    if len(lower_title) < 2 or len(lower_title) > 150:
        return False
    
    c_suite_acronyms = r'\b(ceo|cto|cfo|coo|cmo|cio|cpo|cso|cro|cdo|clo|cao|cbo|cco|chro|cino|cgo|cko|cno|cqo|csco|cvo|cwo|cxo|ciso)\b'
    if re.search(c_suite_acronyms, lower_title):
        return True
    
    normalized_title = re.sub(r'[-–—]', ' ', lower_title)
    normalized_title = re.sub(r'\s+', ' ', normalized_title)
    
    if re.search(r'\bchief[\s]+\w+', normalized_title): return True
    if re.search(r'\b(founder|co\s*-?\s*founder)\b', normalized_title): return True
    if re.search(r'\b(owner|co\s*-?\s*owner)\b', normalized_title): return True
    if re.search(r'\bpresident\b', normalized_title): return True
    if re.search(r'\bvice[\s]+president\b', normalized_title): return True
    if re.search(r'\b(svp|evp|avp)\b', normalized_title): return True
    if re.search(r'^vp\b', normalized_title) or re.search(r'\bvp[\s,]', normalized_title) or re.search(r'\bvp\s+of\b', normalized_title): return True
    if re.search(r'\bchair(man|woman|person)?\b', normalized_title): return True
    if re.search(r'\bboard[\s]+(member|of[\s]+directors)\b', normalized_title): return True
    if re.search(r'\bmanaging[\s]+director\b', normalized_title): return True
    if re.search(r'\bexecutive[\s]+director\b', normalized_title): return True
    if re.search(r'\bgeneral[\s]+counsel\b', normalized_title): return True
    if re.search(r'\bgeneral[\s]+manager\b', normalized_title): return True
    if re.search(r'\bmanaging[\s]+partner\b', normalized_title): return True
    
    return False

def is_valid_person_name(text):
    if not text or len(text) < 4 or len(text) > 50:
        return False
    
    trimmed = text.strip()
    
    if re.search(r'[@#$%^&*()+=\[\]{}|\\<>!?]', trimmed): return False
    if re.match(r'^\d', trimmed): return False
    if re.search(r'\.(com|org|net|io|ai|co)$', trimmed, re.IGNORECASE): return False
    
    lower_text = trimmed.lower()
    
    common_non_name_words = [
        "about", "above", "across", "after", "again", "against", "all", "also", "always", "an", "and", "any", "are",
        "asked", "at", "be", "been", "before", "being", "below", "between", "big", "both", "but", "by", "can",
        "click", "come", "could", "create", "do", "does", "done", "down", "during", "each", "easy", "even",
        "every", "few", "find", "first", "for", "frequently", "from", "get", "give", "go", "good", "great",
        "had", "has", "have", "having", "he", "help", "helping", "her", "here", "him", "his", "how",
        "if", "in", "into", "is", "it", "its", "just", "know", "last", "learn", "like", "long", "look",
        "made", "make", "many", "may", "me", "more", "most", "much", "must", "my", "need", "new", "no",
        "not", "now", "of", "off", "on", "one", "only", "or", "other", "our", "out", "over", "overview",
        "own", "professional", "questions", "quick", "read", "right", "same", "say", "see", "she", "should",
        "small", "so", "some", "such", "take", "teams", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "think", "this", "those", "through", "to", "too", "under", "up", "us",
        "use", "very", "want", "was", "way", "we", "well", "were", "what", "when", "where", "which",
        "while", "who", "why", "will", "with", "would", "you", "your", "businesses", "companies", "products",
        "platform", "software", "marketing", "sales", "support", "customer", "customers", "clients", "partners",
        "features", "benefits", "pricing", "resources", "blog", "news", "events", "webinar", "demo", "trial",
        "free", "get started", "sign up", "login", "register", "subscribe", "download", "watch", "view",
        "faq", "faqs", "onboarding", "integration", "integrations", "automation", "workflow", "workflows",
        "email", "unlimited", "contacts", "leads", "campaigns", "analytics", "reports", "reporting", "dashboard",
        "api", "tools", "tool", "apps", "app", "mobile", "desktop", "web", "cloud", "online", "social",
        "media", "content", "seo", "ppc", "crm", "erp", "saas", "b2b", "b2c", "roi", "kpi", "kpis",
        "advanced", "basic", "premium", "enterprise", "starter", "pro", "plus", "lite", "standard",
        "powerful", "simple", "smart", "fast", "easy", "secure", "reliable", "scalable", "flexible",
        "custom", "personalized", "automated", "real-time", "realtime", "next-gen", "nextgen", "ai-powered",
        "landing", "page", "pages", "home", "homepage", "site", "website", "builder", "template", "templates",
        "form", "forms", "popup", "popups", "widget", "widgets", "header", "footer", "sidebar", "navigation",
        "button", "buttons", "link", "links", "image", "images", "video", "videos", "audio", "file", "files"
    ]
    
    words = lower_text.split()
    common_word_count = sum(1 for w in words if w in common_non_name_words)
    if common_word_count >= 1: return False
    
    non_name_patterns = [
        r'\bjobs?\b', r'\bcareers?\b', r'\bcontact\b', r'\brelations\b', r'\binvestor\b',
        r'\bagenda\b', r'\binsights?\b', r'\bexplore\b', r'\brelated\b', r'\bdesign\b',
        r'\bgrowth\b', r'\banalysis\b', r'\bservices?\b', r'\bsolutions?\b', r'\bstrategy\b',
        r'\bunit\b', r'\bbusiness\b', r'\banalytics?\b', r'\bprogram\b', r'\bmanager\b',
        r'\bconsulting\b', r'\badvisory\b', r'\baudit\b', r'\btax\b', r'\brisk\b',
        r'\bdata\b', r'\bdigital\b', r'\btechnology\b', r'\binnovation\b', r'\btransformation\b',
        r'\bcorporate\b', r'\bglobal\b', r'\bamericas\b', r'\bemea\b', r'\bapac\b',
        r'\bdepartment\b', r'\bdivision\b', r'\bgroup\b', r'\bteam\b', r'\bstaff\b',
        r'\bboard\b', r'\bmember\b', r'\bexecutive\b', r'\bleadership\b', r'\bmanagement\b',
        r'\blearn\s+more\b', r'\bread\s+more\b', r'\bget\s+started\b', r'\bsign\s+up\b',
        r'\bour\s+\w+\b', r'\byour\s+\w+\b', r'\bwhat\s+\w+\b', r'\bhow\s+\w+\b', r'\bwhy\s+\w+\b'
    ]
    
    for pattern in non_name_patterns:
        if re.search(pattern, lower_text): return False
    
    normalized = re.sub(r',?\s*(Jr\.?|Sr\.?|III|II|IV|PhD|MD|MBA|Esq\.?)$', '', trimmed, flags=re.IGNORECASE).strip()
    name_words = normalized.split()
    
    if len(name_words) < 2 or len(name_words) > 4: return False
    
    def is_valid_name_word(word):
        if len(word) < 2: return False
        if re.match(r'^[A-Z][a-z]+$', word): return True
        if re.match(r'^[A-Z]\.$', word): return True
        if re.match(r'^(de|van|von|la|le|del|di|da|das|dos|du|el|al|bin|ibn)$', word, re.IGNORECASE): return True
        if re.match(r"^O'[A-Z][a-z]+$", word, re.IGNORECASE): return True
        if re.match(r"^Mc[A-Z][a-z]+$", word, re.IGNORECASE): return True
        if re.match(r"^Mac[A-Z][a-z]+$", word, re.IGNORECASE): return True
        if re.match(r"^[A-Z][a-z]+-[A-Z][a-z]+$", word): return True
        return False
    
    valid_name_words = [w for w in name_words if is_valid_name_word(w)]
    if len(valid_name_words) < 2: return False
    
    if not is_valid_name_word(name_words[0]) or not is_valid_name_word(name_words[-1]): return False
    
    return True

def clean_title(title):
    return re.sub(r'\s+', ' ', title).strip()[:150]

def get_domain_from_url(url):
    try:
        parsed = urlparse(url)
        return parsed.hostname.replace("www.", "")
    except:
        return ""

def get_base_url(url):
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    except:
        return url

def extract_company_name(url, soup):
    og_site_name = soup.find('meta', property='og:site_name')
    if og_site_name and og_site_name.get('content'):
        return og_site_name['content'].strip()
    
    title = soup.title.string if soup.title else ""
    if title:
        parts = re.split(r'[|\-–—]', title)
        if len(parts) > 0:
            return parts[-1].strip()
    
    try:
        parsed = urlparse(url)
        return parsed.hostname.replace("www.", "").split(".")[0]
    except:
        return "Unknown Company"

def discover_sitemap(base_url):
    sitemap_locations = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
        f"{base_url}/sitemap-index.xml",
        f"{base_url}/sitemaps.xml",
    ]
    
    try:
        robots_result = fetch_page(f"{base_url}/robots.txt")
        if robots_result and robots_result['html']:
            sitemap_matches = re.findall(r'Sitemap:\s*(https?://[^\s]+)', robots_result['html'], re.IGNORECASE)
            for match in sitemap_matches:
                url = match.strip()
                if url not in sitemap_locations:
                    sitemap_locations.insert(0, url)
    except Exception as e:
        logger.debug(f"Could not fetch robots.txt: {e}")
        
    all_urls = []
    
    for sitemap_url in sitemap_locations:
        try:
            result = fetch_page(sitemap_url)
            if not result or not result['html']: continue
            
            soup = BeautifulSoup(result['html'], 'xml')
            
            # Check if index
            sitemap_tags = soup.find_all('sitemap')
            if sitemap_tags:
                for sitemap_tag in sitemap_tags:
                    loc_tag = sitemap_tag.find('loc')
                    if loc_tag:
                        child_url = loc_tag.text.strip()
                        # Limit to first 3 child sitemaps to avoid explosion
                        if len(all_urls) > 1000: break 
                        
                        try:
                            child_result = fetch_page(child_url)
                            if child_result and child_result['html']:
                                child_soup = BeautifulSoup(child_result['html'], 'xml')
                                url_tags = child_soup.find_all('url')
                                for url_tag in url_tags:
                                    loc = url_tag.find('loc')
                                    if loc:
                                        all_urls.append(loc.text.strip())
                        except:
                            continue
            
            # Direct URL entries
            url_tags = soup.find_all('url')
            for url_tag in url_tags:
                loc = url_tag.find('loc')
                if loc:
                    all_urls.append(loc.text.strip())
            
            if all_urls: break
        except Exception as e:
            continue
            
    return all_urls

def score_url_for_executives(url):
    url_lower = url.lower()
    score = 0
    
    high_value = ["leadership", "executive", "management", "board", "directors", "founders"]
    medium_value = ["about-us", "about", "company", "who-we-are"]
    low_value = ["team", "people", "staff", "our-team", "meet-the-team", "employees"]
    negative = [
        "blog", "news", "product", "shop", "cart", "checkout", "login",
        "signup", "register", "privacy", "terms", "legal", "faq", "help",
        "support", "careers", "jobs", "press", "media", "investor",
        "category", "tag", "archive", "page/", "/p/", "?", "#",
        "contact", "contact-us"
    ]
    
    for kw in high_value:
        if kw in url_lower: score += 15
    for kw in medium_value:
        if kw in url_lower: score += 5
    for kw in low_value:
        if kw in url_lower: score += 2
    for kw in negative:
        if kw in url_lower: score -= 20
        
    path_depth = len(url.split('/')) - 3
    if path_depth <= 2: score += 3
    if path_depth <= 1: score += 2
    
    return score

def find_executive_pages(base_url, soup):
    logger.info(f"Discovering sitemap for {base_url}...")
    
    sitemap_urls = discover_sitemap(base_url)
    
    if sitemap_urls:
        logger.info(f"Found {len(sitemap_urls)} URLs in sitemap")
        scored_urls = []
        for url in sitemap_urls:
            score = score_url_for_executives(url)
            if score > 0:
                scored_urls.append({'url': url, 'score': score})
        
        scored_urls.sort(key=lambda x: x['score'], reverse=True)
        top_urls = [item['url'] for item in scored_urls[:5]]
        logger.info(f"Selected {len(top_urls)} high-value pages from sitemap")
        return top_urls
        
    logger.info("No sitemap found, falling back to link analysis...")
    
    executive_keywords = ["leadership", "management", "executives", "board", "founders", "about"]
    links = []
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text().lower()
        href_lower = href.lower()
        
        is_exec_link = any(kw in href_lower or kw in text for kw in executive_keywords)
        
        if is_exec_link:
            full_url = urljoin(base_url, href)
            
            if any(x in full_url for x in ["mailto:", "tel:", "javascript:", "#"]): continue
            
            if full_url not in links:
                links.append(full_url)
                
    scored_links = []
    for url in links:
        score = score_url_for_executives(url)
        if score > 0:
            scored_links.append({'url': url, 'score': score})
            
    scored_links.sort(key=lambda x: x['score'], reverse=True)
    return [item['url'] for item in scored_links[:5]]

def extract_people_from_page(soup, page_url):
    people = []
    seen_names = set()
    
    person_container_selectors = [
        ".team-member", ".executive", ".leadership-member", ".person", ".staff-member",
        ".leader", ".bio", ".profile", ".employee", ".member", ".team-card",
        '[class*="team-member"]', '[class*="executive"]', '[class*="leadership"]',
        '[class*="person"]', '[class*="bio"]', '[class*="profile"]',
        '[itemtype*="Person"]', '[data-person]', '[data-team-member]',
        ".card", "article", "li.team", "div.team", ".grid > div", ".flex > div"
    ]
    
    # Combine selectors
    # BeautifulSoup doesn't support all CSS selectors perfectly like jQuery/Cheerio, 
    # so we iterate and try to find elements.
    
    containers = []
    for selector in person_container_selectors:
        try:
            found = soup.select(selector)
            containers.extend(found)
        except:
            pass
            
    # Remove duplicates (by object identity)
    containers = list(set(containers))
    
    for container in containers:
        person_name = ""
        person_title = ""
        
        # Check headings
        for heading in container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = heading.get_text().strip()
            if text and len(text) < 150:
                if is_valid_person_name(text) and not is_executive_title(text) and not person_name:
                    person_name = text
                elif is_executive_title(text) and not person_title:
                    person_title = clean_title(text)
                    
        # Check specific name/title classes if not found
        if not person_name:
            name_selectors = [
                ".name", ".person-name", ".team-name", ".member-name", ".executive-name",
                '[class*="name"]', '[itemprop="name"]', "strong", "b", "a"
            ]
            for sel in name_selectors:
                try:
                    els = container.select(sel)
                    for el in els:
                        text = el.get_text().strip()
                        if is_valid_person_name(text) and not is_executive_title(text):
                            person_name = text
                            break
                    if person_name: break
                except: pass
                
        if not person_title:
            title_selectors = [
                ".title", ".position", ".role", ".job-title", ".designation", ".job",
                '[class*="title"]', '[class*="position"]', '[class*="role"]',
                '[itemprop="jobTitle"]', "p", "span", "em", "small"
            ]
            for sel in title_selectors:
                try:
                    els = container.select(sel)
                    for el in els:
                        text = el.get_text().strip()
                        if text and is_executive_title(text) and len(text) < 150 and text != person_name:
                            person_title = clean_title(text)
                            break
                    if person_title: break
                except: pass
        
        # Fallback to text analysis
        if not person_name or not person_title:
            lines = [l.strip() for l in container.get_text().split('\n') if l.strip() and len(l) < 150]
            for line in lines:
                if (not person_name or not person_title) and re.search(r'[-–—:,|]', line):
                    parts = [p.strip() for p in re.split(r'\s*[-–—:,|]\s*', line) if p.strip()]
                    
                    for part in parts:
                        if not person_name and is_valid_person_name(part) and not is_executive_title(part):
                            person_name = part
                        if not person_title and is_executive_title(part):
                            person_title = clean_title(part)
                            
                    if not person_name and not person_title and len(parts) >= 2:
                        first = parts[0]
                        rest = " ".join(parts[1:])
                        if is_valid_person_name(first) and not is_executive_title(first):
                            person_name = first
                            if is_executive_title(rest):
                                person_title = clean_title(rest)
                        elif is_executive_title(first) and is_valid_person_name(rest) and not is_executive_title(rest):
                            person_title = clean_title(first)
                            person_name = rest
                            
                if not person_name and is_valid_person_name(line) and not is_executive_title(line):
                    person_name = line
                if not person_title and is_executive_title(line) and line != person_name:
                    person_title = clean_title(line)
                    
        if person_name and person_title and is_executive_title(person_title):
            if person_name.lower() not in seen_names:
                seen_names.add(person_name.lower())
                
                # Extract emails from container
                container_html = str(container)
                emails = extract_emails_from_text(container_html)
                
                people.append({
                    'name': person_name,
                    'title': person_title,
                    'email': emails[0] if emails else None
                })

    # Fallback: Scan body text if no structured people found
    if not people:
        lines = [l.strip() for l in soup.get_text().split('\n') if l.strip()]
        for i in range(len(lines) - 1):
            line = lines[i]
            next_line = lines[i+1]
            
            if is_valid_person_name(line) and is_executive_title(next_line):
                name_lower = line.lower()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    people.append({
                        'name': line,
                        'title': clean_title(next_line),
                        'email': None
                    })
                    
    return people

def extract_emails_from_text(text):
    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(email_regex, text)
    unique_emails = []
    seen = set()
    
    for email in matches:
        if email in seen: continue
        
        # Filter junk
        if any(x in email for x in ["example.com", "test.com", "placeholder", "email.com"]): continue
        if any(email.endswith(x) for x in [".png", ".jpg", ".svg", ".gif"]): continue
        if any(email.startswith(x) for x in ["noreply@", "no-reply@", "donotreply@"]): continue
        
        seen.add(email)
        unique_emails.append(email)
        
    return unique_emails

def extract_all_emails_from_page(soup, page_url, company_domain):
    results = []
    seen_emails = set()
    html = str(soup)
    
    all_emails = extract_emails_from_text(html)
    
    department_prefixes = [
        "info@", "contact@", "support@", "hello@", "sales@", "help@",
        "admin@", "careers@", "jobs@", "hr@", "press@", "media@", 
        "marketing@", "team@", "office@", "general@", "feedback@"
    ]
    
    skip_prefixes = ["noreply@", "no-reply@", "donotreply@", "webmaster@"]
    
    for email in all_emails:
        email_lower = email.lower()
        if email_lower in seen_emails: continue
        
        if any(email_lower.startswith(p) for p in skip_prefixes): continue
        
        seen_emails.add(email_lower)
        
        associated_name = None
        associated_title = None
        contact_type = "general"
        
        if any(email_lower.startswith(p) for p in department_prefixes):
            contact_type = "department"
            
        # Try to extract name from email
        email_local = email_lower.split("@")[0].replace(".", " ").replace("_", " ")
        name_parts = [p for p in email_local.split() if len(p) > 1]
        if len(name_parts) >= 2:
            guessed_name = " ".join([p.capitalize() for p in name_parts])
            if is_valid_person_name(guessed_name):
                associated_name = guessed_name
                
        # Search nearby text
        esc_email = re.escape(email)
        nearby_pattern = f"([^<>]{{0,200}}){esc_email}([^<>]{{0,200}})"
        matches = re.findall(nearby_pattern, html, re.IGNORECASE)
        
        if matches:
            for pre, post in matches:
                context = (pre + " " + post).replace("\n", " ")
                context = re.sub(r'<[^>]+>', ' ', context)
                words = context.split()
                
                for i in range(len(words) - 1):
                    two_words = words[i] + " " + words[i+1]
                    if is_valid_person_name(two_words) and not is_executive_title(two_words):
                        associated_name = two_words
                        break
                
                for word in words:
                    # Check word and surrounding phrase
                    try:
                        idx = words.index(word)
                        phrase = " ".join(words[idx:idx+3])
                        if is_executive_title(word) or is_executive_title(phrase):
                            associated_title = clean_title(phrase)
                            contact_type = "executive"
                            break
                    except: pass
                    
        results.append({
            'email': email,
            'name': associated_name,
            'title': associated_title,
            'contact_type': contact_type
        })
        
    return results

def scrape_company_contacts(url):
    base_url = get_base_url(url)
    contacts = []
    scraped_pages = set()
    seen_emails = set()
    seen_names = set()
    
    logger.info(f"Fetching main page: {url}")
    main_page = fetch_page(url)
    if not main_page:
        logger.error(f"Failed to fetch main page: {url}")
        return {'executives': [], 'company_name': "Unknown"}
        
    soup = BeautifulSoup(main_page['html'], 'html.parser')
    company_name = extract_company_name(url, soup)
    company_domain = get_domain_from_url(base_url)
    scraped_pages.add(url)
    
    logger.info(f"Scraping {url} for contacts (domain: {company_domain})")
    
    # Extract from main page
    main_people = extract_people_from_page(soup, url)
    for person in main_people:
        if person['name'].lower() not in seen_names:
            seen_names.add(person['name'].lower())
            if person['email']: seen_emails.add(person['email'].lower())
            is_exec = person['title'] and is_executive_title(person['title'])
            contacts.append({
                'company_url': base_url,
                'company_name': company_name,
                'name': person['name'],
                'title': person['title'] or "Contact",
                'email': person['email'],
                'source_url': url,
                'contact_type': "executive" if is_exec else "general"
            })
            
    main_emails = extract_all_emails_from_page(soup, url, company_domain)
    for email_data in main_emails:
        if email_data['email'].lower() not in seen_emails:
            seen_emails.add(email_data['email'].lower())
            name = email_data['name'] or email_data['email'].split("@")[0]
            contacts.append({
                'company_url': base_url,
                'company_name': company_name,
                'name': name,
                'title': email_data['title'] or "Contact",
                'email': email_data['email'],
                'source_url': url,
                'contact_type': email_data['contact_type']
            })
            
    # Find executive pages
    executive_pages = find_executive_pages(base_url, soup)
    
    pages_to_scrape = [p for p in executive_pages if p not in scraped_pages]
    for p in pages_to_scrape: scraped_pages.add(p)
    
    # Scrape subpages
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(fetch_page, page_url): page_url for page_url in pages_to_scrape}
        
        for future in concurrent.futures.as_completed(future_to_url):
            page_url = future_to_url[future]
            try:
                page_result = future.result()
                if not page_result: continue
                
                page_soup = BeautifulSoup(page_result['html'], 'html.parser')
                page_people = extract_people_from_page(page_soup, page_url)
                page_emails = extract_all_emails_from_page(page_soup, page_url, company_domain)
                
                for person in page_people:
                    if person['name'].lower() not in seen_names:
                        seen_names.add(person['name'].lower())
                        if person['email']: seen_emails.add(person['email'].lower())
                        is_exec = person['title'] and is_executive_title(person['title'])
                        contacts.append({
                            'company_url': base_url,
                            'company_name': company_name,
                            'name': person['name'],
                            'title': person['title'] or "Contact",
                            'email': person['email'],
                            'source_url': page_url,
                            'contact_type': "executive" if is_exec else "general"
                        })
                        
                for email_data in page_emails:
                    if email_data['email'].lower() not in seen_emails:
                        seen_emails.add(email_data['email'].lower())
                        name = email_data['name'] or email_data['email'].split("@")[0]
                        contacts.append({
                            'company_url': base_url,
                            'company_name': company_name,
                            'name': name,
                            'title': email_data['title'] or "Contact",
                            'email': email_data['email'],
                            'source_url': page_url,
                            'contact_type': email_data['contact_type']
                        })
                        
            except Exception as exc:
                logger.error(f'{page_url} generated an exception: {exc}')
                
    logger.info(f"Found {len(contacts)} contacts for {company_name}")
    return {'executives': contacts, 'company_name': company_name}

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return
        
    if 'Website' not in df.columns and 'Email' not in df.columns:
        logger.error("CSV must contain a 'Website' or 'Email' column")
        return
        
    all_contacts = []
    
    for index, row in df.iterrows():
        url = None
        if 'Website' in df.columns and pd.notna(row['Website']):
            url = row['Website']
        elif 'Email' in df.columns and pd.notna(row['Email']):
            email = row['Email']
            if '@' in email:
                domain = email.split('@')[1]
                url = f"https://www.{domain}"
                
        if not url: continue
        
        if not url.startswith('http'):
            url = 'https://' + url
            
        logger.info(f"Processing {index+1}/{len(df)}: {url}")
        result = scrape_company_contacts(url)
        all_contacts.extend(result['executives'])
        
    # Save results
    if all_contacts:
        results_df = pd.DataFrame(all_contacts)
        output_file = file_path.replace('.csv', '_results.csv')
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    else:
        logger.info("No contacts found.")

if __name__ == "__main__":
    # Default test file
    test_file = r"C:\Codespace\DataEnhancement\Websites_Needs_ContactInfo.csv"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
    if os.path.exists(test_file):
        logger.info(f"Starting processing for {test_file}")
        process_csv(test_file)
    else:
        logger.error(f"File not found: {test_file}")
