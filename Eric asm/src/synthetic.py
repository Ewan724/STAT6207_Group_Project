"""Deterministic synthetic demo corpus.

This module exists because the assignment's ``DATA/`` directory is not present
in this execution environment. It writes a corpus with exactly the same folder
structure, file names, PDF headers and marker strings as the real data, so the
full pipeline (PDF parsing, marker splitting, extraction, clustering, AE,
retrieval, metrics, website) can be executed and verified end to end.

Every artifact derived from this corpus is labelled ``synthetic_demo``. Nothing
here is a result on the real assignment data.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from .data import ROLE_FAMILIES

FAMILY_SKILLS: dict[str, list[str]] = {
    "Finance & Legal": [
        "financial reporting", "taxation", "cash flow management", "audit", "budgeting",
        "forecasting", "cost management", "IFRS", "internal controls", "manufacturing costing",
        "ERP", "Excel", "Power BI", "FP&A", "P&L", "compliance", "risk management", "SAP",
        "financial analysis", "variance analysis", "US GAAP",
    ],
    "Sales & BD": [
        "B2B sales", "account management", "CRM", "lead generation", "negotiation",
        "sales pipeline", "key account management", "business development", "cold calling",
        "customer success", "sales forecasting", "technical sales", "Salesforce", "HubSpot",
        "customer relationship management", "B2C sales", "channel management",
    ],
    "Marketing": [
        "SEO", "SEM", "Google Analytics", "content marketing", "social media marketing",
        "email marketing", "marketing automation", "brand management", "campaign management",
        "Go-to-Market", "B2B", "B2C", "copywriting", "Google Ads", "Meta Ads", "CRM",
        "HubSpot", "market research", "demand generation",
    ],
    "Data & AI": [
        "Python", "SQL", "ETL", "ELT", "Airflow", "Spark", "Kafka", "dbt", "Snowflake",
        "AWS", "Google Cloud Platform", "Microsoft Azure", "Machine Learning", "Deep Learning",
        "Natural Language Processing", "Large Language Model", "Retrieval-Augmented Generation",
        "TensorFlow", "PyTorch", "scikit-learn", "Docker", "Kubernetes", "Tableau", "Power BI",
        "data warehousing", "statistics",
    ],
    "Product": [
        "product management", "product strategy", "roadmap", "user research", "Agile", "Scrum",
        "JIRA", "A/B testing", "SQL", "stakeholder management", "WMS", "OMS",
        "product analytics", "Figma", "API", "product discovery", "OKR",
    ],
    "Engineering": [
        "Python", "Java", "JavaScript", "TypeScript", "React", "Node.js", "Go", "C++", "C#",
        ".NET", "Docker", "Kubernetes", "AWS", "Microsoft Azure", "Google Cloud Platform",
        "CI/CD", "DevOps", "microservices", "REST API", "PostgreSQL", "MySQL", "MongoDB",
        "Redis", "Kafka", "Terraform", "Linux", "Git", "Spring Boot",
        "field service", "preventive maintenance", "troubleshooting", "semiconductor equipment",
        "cleanroom", "electrical systems", "PLC", "technical documentation",
    ],
    "Design & Creative": [
        "UI", "UX", "Figma", "Sketch", "Adobe XD", "prototyping", "user research",
        "design systems", "wireframing", "usability testing", "interaction design",
        "visual design", "Adobe Photoshop", "Adobe Illustrator", "HTML", "CSS", "accessibility",
        "motion design", "brand identity",
    ],
    "Human Resources": [
        "talent acquisition", "recruitment", "onboarding", "HRIS", "payroll",
        "employee relations", "performance management", "compensation and benefits",
        "HR policies", "labor law", "Workday", "SAP SuccessFactors", "employer branding",
        "workforce planning",
    ],
}

TITLES = {
    "Finance & Legal": ["Finance Manager", "Regional Finance Controller", "Senior Financial Analyst",
                        "Accounting Manager", "Finance Business Partner", "Internal Audit Manager",
                        "FP&A Manager", "Credit Controller", "Regional Finance Manager"],
    "Sales & BD": ["Sales Manager", "Business Development Manager", "Account Executive",
                   "Key Account Manager", "Technical Sales Specialist", "Regional Sales Director",
                   "Customer Success Manager"],
    "Marketing": ["Marketing Manager", "Growth Marketing Manager", "Digital Marketing Specialist",
                  "Brand Manager", "Content Marketing Manager", "Head of Growth & Digital Transformation",
                  "Campaign Manager", "Go-to-Market Manager"],
    "Data & AI": ["Data Engineer", "Data Scientist", "Machine Learning Engineer", "Analytics Engineer",
                  "Data Analyst", "AI Engineer", "Principal, Data Science"],
    "Product": ["Product Manager", "Senior Product Manager", "Product Owner", "Head of Product",
                "Technical Product Manager"],
    "Engineering": ["Software Engineer", "Backend Developer", "Frontend Developer", "DevOps Engineer",
                    "Tech Lead", "Field Service Engineer", "Full Stack Developer",
                    "Site Reliability Engineer", "Principal Engineer"],
    "Design & Creative": ["UI/UX Designer", "Lead, UI/UX", "Product Designer", "Visual Designer",
                          "UX Researcher", "Design Lead", "Senior Interaction Designer"],
    "Human Resources": ["HR Manager", "Talent Acquisition Specialist", "HR Business Partner",
                        "Recruiter", "People Operations Manager",
                        "Compensation and Benefits Manager"],
}

COMPANIES = [
    "Sea - Shopee", "Grab", "Gojek", "DBS Bank", "Maybank", "OCBC Bank", "AIA", "Prudential",
    "Accenture", "Deloitte", "PwC", "EY", "KPMG", "IBM", "Microsoft", "Google", "Amazon",
    "ShopBack", "Carousell", "Traveloka", "Tokopedia", "Lazada", "GHL Systems", "Silverlake",
    "Hartalega", "Top Glove", "Petronas", "Sime Darby", "Astro", "Maxis", "CelcomDigi",
    "HKT Enterprise Solutions", "PCCW", "Jardine Matheson", "Swire Group", "ASM WPE",
    "Tokyo Electron", "Infineon", "Micron", "GlobalFoundries", "CBTW", "FPT Software",
    "VNG Corporation", "VinGroup", "Siam Commercial Bank", "Kasikornbank", "Bank Mandiri",
    "Telkom Indonesia", "My E.G. Services Bhd", "Step Ahead", "Voyage Asia",
]

UNIVERSITIES = [
    "University of Malaya", "National University of Singapore", "Nanyang Technological University",
    "Universiti Teknologi Malaysia", "Universitas Indonesia", "Institut Teknologi Bandung",
    "Chulalongkorn University", "University of Hong Kong", "Chinese University of Hong Kong",
    "Vietnam National University", "De La Salle University", "University of the Philippines",
    "Monash University", "University of Melbourne", "Imperial College London",
]

COUNTRIES = ["Malaysia", "Singapore", "Indonesia", "Vietnam", "Thailand", "Philippines",
             "Hong Kong", "India"]

DEGREES = ["Bachelor's Degree", "Master's Degree", "Bachelor's Degree", "Bachelor's Degree",
           "Master's Degree", "Diploma"]

FIELDS = {
    "Finance & Legal": ["Finance", "Accounting", "Economics", "Business Administration"],
    "Sales & BD": ["Business Administration", "Marketing", "Economics", "Engineering"],
    "Marketing": ["Marketing", "Communications", "Business Administration", "Digital Media"],
    "Data & AI": ["Computer Science", "Data Science", "Statistics", "Mathematics", "Engineering"],
    "Product": ["Computer Science", "Business Administration", "Information Systems", "Design"],
    "Engineering": ["Computer Science", "Electrical Engineering", "Software Engineering",
                    "Mechanical Engineering", "Information Technology"],
    "Design & Creative": ["Design", "Visual Communication", "Interaction Design", "Fine Arts"],
    "Human Resources": ["Human Resources", "Psychology", "Business Administration", "Sociology"],
}

LANGUAGES = ["English", "Malay", "Mandarin", "Indonesian", "Thai", "Vietnamese", "Tagalog", "Hindi"]


# --------------------------------------------------------------------------- #
# Specific profiles (hard lookalikes and hired candidates)
# --------------------------------------------------------------------------- #
SPECIAL_PROFILES: dict[str, dict] = {
    "CV-002": dict(title="Field Service Engineer", company="ASM WPE", family="Engineering",
                   country="Singapore", years=6, seniority="mid",
                   skills=["field service", "preventive maintenance", "troubleshooting",
                           "semiconductor equipment", "cleanroom", "electrical systems", "PLC",
                           "technical documentation", "customer support", "Python", "SQL"]),
    "CV-012": dict(title="Data Engineer", company="Grab", family="Data & AI",
                   country="Singapore", years=5, seniority="mid",
                   skills=["Python", "SQL", "ETL", "Airflow", "Spark", "AWS", "Kafka",
                           "data warehousing", "Docker", "dbt", "Tableau"]),
    "CV-018": dict(title="Tech Lead", company="GHL Systems", family="Engineering",
                   country="Malaysia", years=9, seniority="lead",
                   skills=["Java", "Spring Boot", "microservices", "AWS", "Docker", "Kubernetes",
                           "CI/CD", "REST API", "PostgreSQL", "mentoring", "Linux"]),
    "CV-042": dict(title="Principal, Data Science", company="Accenture", family="Data & AI",
                   country="Singapore", years=10, seniority="lead",
                   skills=["Python", "Machine Learning", "Natural Language Processing", "SQL",
                           "statistics", "PyTorch", "stakeholder management", "data visualization",
                           "Large Language Model", "Docker"]),
    "CV-045": dict(title="Data Engineer", company="CBTW", family="Data & AI",
                   country="Malaysia", years=4, seniority="mid",
                   skills=["Python", "SQL", "ETL", "Airflow", "Spark", "Microsoft Azure", "Kafka",
                           "dbt", "data warehousing", "Docker", "Git"]),
    "CV-048": dict(title="Lead, UI/UX", company="Thumb Technologies", family="Design & Creative",
                   country="Malaysia", years=7, seniority="lead",
                   skills=["UI", "UX", "Figma", "design systems", "prototyping", "user research",
                           "wireframing", "usability testing", "interaction design", "accessibility",
                           "HTML", "CSS"]),
    "CV-058": dict(title="Technical Sales Specialist", company="PT. Sinar Mas", family="Sales & BD",
                   country="Indonesia", years=6, seniority="mid",
                   skills=["B2B sales", "account management", "CRM", "lead generation", "negotiation",
                           "sales pipeline", "technical sales", "forecasting", "Salesforce", "Excel"]),
    "CV-069": dict(title="Head of Growth & Digital Transformation", company="Voyage Asia",
                   family="Marketing", country="Singapore", years=10, seniority="lead",
                   skills=["Go-to-Market", "SEO", "SEM", "marketing automation", "Google Analytics",
                           "content marketing", "B2B", "CRM", "campaign management", "HubSpot",
                           "Google Ads"]),
    "CV-079": dict(title="Regional Finance Manager", company="Hartalega", family="Finance & Legal",
                   country="Malaysia", years=14, seniority="lead",
                   skills=["financial reporting", "taxation", "cash flow management", "audit",
                           "budgeting", "forecasting", "cost management", "IFRS", "internal controls",
                           "manufacturing costing", "ERP", "Excel", "SAP"]),
    "CV-097": dict(title="Senior Product Manager", company="Grab", family="Product",
                   country="Singapore", years=7, seniority="senior",
                   skills=["product management", "roadmap", "WMS", "OMS", "Agile", "Scrum", "JIRA",
                           "SQL", "stakeholder management", "product analytics", "Figma"]),
}

HIRED_PROFILES = [
    dict(jd_id="P01", family="Finance & Legal", jd_title="Regional Finance Controller",
         title="Regional Finance Manager", company="Cardia Bioplastics (Malaysia) Sdn Bhd",
         country="Malaysia", years=16, seniority="lead",
         location="Kuala Lumpur, Malaysia", salary="MYR 240,000 - 300,000 per annum",
         min_years=15, remote="no",
         must=["financial reporting", "taxation", "cash flow management", "audit", "budgeting",
               "forecasting", "cost management", "IFRS", "internal controls", "manufacturing costing"],
         nice=["ERP", "Excel", "PowerPoint", "SAP"],
         responsibilities=["Lead the finance function for Malaysia and Thailand",
                           "Own financial reporting, taxation, budgeting and cash flow",
                           "Partner with manufacturing operations on costing and cost control"],
         skills=["financial reporting", "taxation", "cash flow management", "audit", "budgeting",
                 "forecasting", "cost management", "IFRS", "internal controls", "manufacturing costing",
                 "ERP", "Excel", "SAP", "P&L", "stakeholder management"]),
    dict(jd_id="P02", family="Sales & BD", jd_title="Sales Manager in Indonesia",
         title="Technical Sales Specialist", company="PT. Chemico Surabaya",
         country="Indonesia", years=7, seniority="mid",
         location="Surabaya, Indonesia", salary="IDR 300,000,000 - 420,000,000 per annum",
         min_years=5, remote="no",
         must=["B2B sales", "account management", "CRM", "lead generation", "negotiation",
               "sales pipeline", "technical sales", "forecasting"],
         nice=["Salesforce", "Excel", "Indonesian language"],
         responsibilities=["Grow the industrial chemicals account base in Indonesia",
                           "Manage the full sales cycle from lead generation to closing",
                           "Build technical proposals with the application engineering team"],
         skills=["B2B sales", "account management", "CRM", "lead generation", "negotiation",
                 "sales pipeline", "technical sales", "forecasting", "Salesforce", "Excel",
                 "customer relationship management", "key account management"]),
    dict(jd_id="P03", family="Marketing", jd_title="GTM Marketing Manager / Lead",
         title="Head of Growth & Digital Transformation", company="Step Ahead",
         country="Singapore", years=11, seniority="lead", location=None, salary=None,
         min_years=8, remote="hybrid",
         must=["Go-to-Market", "SEO", "SEM", "marketing automation", "Google Analytics",
               "content marketing", "B2B", "campaign management", "CRM"],
         nice=["HubSpot", "Google Ads", "market research"],
         responsibilities=["Design and execute the regional go-to-market plan",
                           "Own demand generation across paid, organic and lifecycle channels",
                           "Build the marketing analytics stack and report on funnel performance"],
         skills=["Go-to-Market", "SEO", "SEM", "marketing automation", "Google Analytics",
                 "content marketing", "B2B", "CRM", "campaign management", "HubSpot",
                 "Google Ads", "demand generation"]),
    dict(jd_id="P04", family="Data & AI", jd_title="Data Engineer",
         title="Data Engineer", company="BP", country="Malaysia", years=6, seniority="senior",
         location="Kuala Lumpur, Malaysia", salary="MYR 144,000 - 180,000 per annum",
         min_years=5, remote="hybrid",
         must=["Python", "SQL", "ETL", "Airflow", "Spark", "Microsoft Azure", "Kafka",
               "data warehousing", "Docker", "dbt"],
         nice=["Databricks", "Terraform", "Tableau"],
         responsibilities=["Build and operate batch and streaming data pipelines",
                           "Model curated datasets in the Azure data platform",
                           "Partner with analytics and data science teams on data contracts"],
         skills=["Python", "SQL", "ETL", "Airflow", "Spark", "Microsoft Azure", "Kafka",
                 "data warehousing", "Docker", "dbt", "Git", "Databricks"]),
    dict(jd_id="P05", family="Product", jd_title="Product Manager - WMS & Warehouse Integrations",
         title="Senior Product Manager", company="Shopee International",
         country="Singapore", years=8, seniority="senior",
         location="Singapore", salary="SGD 144,000 - 180,000 per annum",
         min_years=6, remote="yes",
         must=["product management", "roadmap", "WMS", "OMS", "Agile", "Scrum", "JIRA",
               "SQL", "stakeholder management"],
         nice=["product analytics", "Figma", "API"],
         responsibilities=["Own the warehouse management and order management roadmap",
                           "Translate warehouse operator needs into product requirements",
                           "Coordinate rollout with engineering, operations and partners"],
         skills=["product management", "roadmap", "WMS", "OMS", "Agile", "Scrum", "JIRA",
                 "SQL", "stakeholder management", "product analytics", "API", "Figma"]),
    dict(jd_id="P06", family="Data & AI", jd_title="Deployment Strategist",
         title="Principal, Data Science", company="Boston Consulting Group",
         country="Singapore", years=9, seniority="lead",
         location="Singapore", salary="SGD 180,000 - 240,000 per annum",
         min_years=7, remote="hybrid",
         must=["Python", "Machine Learning", "Natural Language Processing", "Large Language Model",
               "SQL", "stakeholder management", "statistics", "data visualization"],
         nice=["PyTorch", "Docker", "Retrieval-Augmented Generation"],
         responsibilities=["Lead client deployments of AI products",
                           "Translate business problems into machine-learning solutions",
                           "Present findings to senior client stakeholders"],
         skills=["Python", "Machine Learning", "Natural Language Processing", "Large Language Model",
                 "SQL", "stakeholder management", "statistics", "data visualization", "PyTorch",
                 "Docker", "Retrieval-Augmented Generation"]),
    dict(jd_id="P07", family="Marketing", jd_title="Marketing Manager (HK & SG)",
         title="Marketing Manager", company="HKT Enterprise Solutions",
         country="Hong Kong", years=9, seniority="senior",
         location="Hong Kong", salary="HKD 600,000 - 720,000 per annum",
         min_years=7, remote="no",
         must=["campaign management", "SEO", "SEM", "Google Analytics", "CRM",
               "content marketing", "B2B", "brand management", "email marketing"],
         nice=["HubSpot", "Google Ads", "marketing automation"],
         responsibilities=["Own marketing campaigns for Hong Kong and Singapore",
                           "Manage agency partners and the regional campaign calendar",
                           "Report pipeline contribution and campaign ROI"],
         skills=["campaign management", "SEO", "SEM", "Google Analytics", "CRM",
                 "content marketing", "B2B", "brand management", "email marketing",
                 "HubSpot", "Google Ads", "marketing automation"]),
    dict(jd_id="P08", family="Engineering", jd_title="Field Service Engineer (FSE)",
         title="Field Service Engineer", company="Tokyo Electron Singapore Pte Ltd",
         country="Singapore", years=5, seniority="mid",
         location="Singapore", salary="SGD 60,000 - 78,000 per annum",
         min_years=3, remote="no",
         must=["field service", "preventive maintenance", "troubleshooting",
               "semiconductor equipment", "cleanroom", "electrical systems", "PLC",
               "technical documentation"],
         nice=["customer support", "Python", "SQL"],
         responsibilities=["Install, maintain and repair semiconductor process equipment",
                           "Perform preventive maintenance in cleanroom environments",
                           "Document service visits and escalate complex faults"],
         skills=["field service", "preventive maintenance", "troubleshooting",
                 "semiconductor equipment", "cleanroom", "electrical systems", "PLC",
                 "technical documentation", "customer support", "Python"]),
    dict(jd_id="P09", family="Engineering", jd_title="Tech Lead",
         title="Tech Lead", company="JaGaApp / Red Ideas Holdings Berhad",
         country="Malaysia", years=10, seniority="lead",
         location="Kuala Lumpur, Malaysia", salary="MYR 180,000 - 216,000 per annum",
         min_years=8, remote="hybrid",
         must=["Java", "Spring Boot", "microservices", "AWS", "Docker", "Kubernetes",
               "CI/CD", "REST API", "PostgreSQL", "mentoring"],
         nice=["Kafka", "Terraform", "JavaScript"],
         responsibilities=["Lead a squad of six engineers delivering the core platform",
                           "Own service architecture, code quality and production reliability",
                           "Mentor engineers and run technical design reviews"],
         skills=["Java", "Spring Boot", "microservices", "AWS", "Docker", "Kubernetes",
                 "CI/CD", "REST API", "PostgreSQL", "mentoring", "Linux", "Terraform"]),
    dict(jd_id="P10", family="Design & Creative", jd_title="Regional Lead Designer UI/UX",
         title="Lead, UI/UX", company="My E.G. Services Bhd",
         country="Malaysia", years=8, seniority="lead",
         location=None, salary="MYR 144,000 - 180,000 per annum",
         min_years=6, remote="hybrid",
         must=["UI", "UX", "Figma", "design systems", "prototyping", "user research",
               "wireframing", "usability testing", "interaction design", "accessibility"],
         nice=["HTML", "CSS", "motion design"],
         responsibilities=["Lead the regional product design team",
                           "Own the design system and interaction patterns",
                           "Run research and usability testing with regional users"],
         skills=["UI", "UX", "Figma", "design systems", "prototyping", "user research",
                 "wireframing", "usability testing", "interaction design", "accessibility",
                 "HTML", "CSS"]),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pick(rng: random.Random, seq: list, k: int) -> list:
    k = min(k, len(seq))
    return rng.sample(seq, k)


def _seniority_for_years(years: int) -> str:
    if years < 2:
        return "junior"
    if years < 5:
        return "mid"
    if years < 9:
        return "senior"
    return "lead"


def _profile(doc_id: str, rng: random.Random) -> dict:
    if doc_id in SPECIAL_PROFILES:
        p = dict(SPECIAL_PROFILES[doc_id])
        p.setdefault("education", None)
        return p
    family = ROLE_FAMILIES[(int(doc_id.split("-")[1]) - 1) % len(ROLE_FAMILIES)]
    years = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14])
    seniority = _seniority_for_years(years)
    title = rng.choice(TITLES[family])
    skills = _pick(rng, FAMILY_SKILLS[family], rng.randint(9, 14))
    # Add a few adjacent-family skills to make the corpus realistic.
    other = rng.choice([f for f in ROLE_FAMILIES if f != family])
    skills += _pick(rng, FAMILY_SKILLS[other], rng.randint(1, 3))
    skills = list(dict.fromkeys(skills))
    return dict(title=title, company=rng.choice(COMPANIES), family=family,
                country=rng.choice(COUNTRIES), years=years, seniority=seniority,
                skills=skills)


def _work_history(rng: random.Random, profile: dict, doc_id: str) -> list[dict]:
    years = profile["years"]
    n_roles = 2 if years < 6 else 3
    remaining = max(1, years - 1)
    spans = []
    for i in range(n_roles - 1):
        length = max(1, remaining // (n_roles - i))
        spans.append(length)
        remaining -= length
    spans.append(max(1, remaining))
    current_year = 2026
    rows = []
    for i, length in enumerate(spans):
        end = current_year - sum(spans[:i])
        start = end - length
        title = profile["title"] if i == 0 else rng.choice(TITLES[profile["family"]])
        company = profile["company"] if i == 0 else rng.choice(COMPANIES)
        rows.append(dict(title=title, company=company, start=f"{start}-{rng.randint(1,12):02d}",
                         end="Present" if i == 0 else f"{end}-{rng.randint(1,12):02d}",
                         length=length))
    return rows


def _cv_lines(doc_id: str, profile: dict, rng: random.Random) -> list[str]:
    work = _work_history(rng, profile, doc_id)
    degree = rng.choice(DEGREES)
    field = rng.choice(FIELDS[profile["family"]])
    uni = rng.choice(UNIVERSITIES)
    lang = rng.sample(LANGUAGES, rng.randint(2, 3))
    certs = []
    if profile["family"] == "Finance & Legal":
        certs = rng.sample(["CPA Australia", "ACCA", "CIMA", "CFA Level II"], 2)
    elif profile["family"] == "Data & AI":
        certs = rng.sample(["Microsoft Certified: Azure Data Engineer", "AWS Certified Data Analytics",
                            "Google Professional Data Engineer", "Databricks Certified Associate"], 1)
    elif profile["family"] == "Engineering":
        certs = rng.sample(["AWS Certified Solutions Architect", "CKA", "PMP"], 1)
    elif profile["family"] == "Design & Creative":
        certs = ["Nielsen Norman Group UX Certification"] if rng.random() > 0.5 else []

    lines = [
        f"CURRENT TITLE: {profile['title']}",
        f"CURRENT COMPANY: {profile['company']}",
        f"ROLE FAMILY: {profile['family']}",
        f"LOCATION: {profile['country']}",
        f"SENIORITY: {profile['seniority']}",
        f"TOTAL YEARS EXPERIENCE: {profile['years']}",
        "",
        "PROFILE",
        f"Experienced {profile['title']} with {profile['years']} years across "
        f"{profile['family'].lower()} roles in {profile['country']}. "
        "Strong track record of delivering measurable outcomes and working with cross-functional teams.",
        "",
        "CORE SKILLS",
        ", ".join(profile["skills"]),
        "",
        "WORK EXPERIENCE",
    ]
    bullets = [
        "Led cross-functional delivery and improved key performance indicators.",
        "Partnered with regional stakeholders on planning, reporting and execution.",
        "Introduced process improvements that reduced turnaround time.",
        "Mentored junior team members and documented operating procedures.",
        "Managed vendor relationships and negotiated commercial terms.",
    ]
    for i, role in enumerate(work):
        lines.append(f"- {role['title']} | {role['company']} | {role['start']} - {role['end']}")
        for b in rng.sample(bullets, 2):
            # A few truncated bullets mimic the real corpus.
            lines.append("  - " + (b[:-9] + "..." if rng.random() < 0.10 else b))
        if i == 0:
            lines.append("  - Owned the regional roadmap and monthly performance review.")
    lines += ["", "EDUCATION", f"- {degree} | {field} | {uni} | 201{rng.randint(0,9)}"]
    if certs:
        lines += ["", "CERTIFICATIONS"] + [f"- {c}" for c in certs]
    lines += ["", "LANGUAGES", ", ".join(f"{l} ({'native' if i == 0 else 'professional'})"
                                        for i, l in enumerate(lang))]
    return lines


def _jd_lines(jd: dict) -> list[str]:
    lines = [
        f"JOB TITLE: {jd['jd_title']}",
        f"ROLE FAMILY: {jd['family']}",
    ]
    if jd.get("location"):
        lines.append(f"LOCATION: {jd['location']}")
    if jd.get("salary"):
        lines.append(f"SALARY: {jd['salary']}")
    lines.append(f"REMOTE: {jd.get('remote', 'unknown')}")
    lines += [
        f"SENIORITY: {jd.get('seniority', 'unknown')}",
        f"MIN YEARS EXPERIENCE: {jd.get('min_years', 0)}",
        "",
        "ABOUT THE ROLE",
        f"We are hiring a {jd['jd_title']} to join our {jd['family']} team. "
        "The role owns delivery across the region and works closely with business stakeholders.",
        "",
        "KEY RESPONSIBILITIES",
    ] + [f"- {r}" for r in jd.get("responsibilities", [])]
    lines += ["", "REQUIREMENTS (MUST-HAVE)"]
    lines += [f"- {s}" for s in jd["must"]]
    if jd.get("nice"):
        lines += ["", "NICE TO HAVE"] + [f"- {s}" for s in jd["nice"]]
    if jd["jd_id"] == "P01":
        lines += ["", "SCOPE: Malaysia and Thailand; regional travel required."]
    if jd["jd_id"] == "P02":
        lines += ["", "Indonesian Citizen | Based in Indonesia. Fluent Indonesian and English required."]
    return lines


def _unlabelled_jds(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed + 7)
    rows = []
    for i in range(1, n + 1):
        family = ROLE_FAMILIES[(i - 1) % len(ROLE_FAMILIES)]
        title = rng.choice(TITLES[family])
        must = _pick(rng, FAMILY_SKILLS[family], rng.randint(5, 8))
        nice = [s for s in FAMILY_SKILLS[family] if s not in must][:rng.randint(2, 4)]
        country = rng.choice(COUNTRIES)
        rows.append(dict(
            num=i, position=title, family=family, country=country,
            city=rng.choice(["Kuala Lumpur", "Singapore", "Jakarta", "Ho Chi Minh City",
                             "Bangkok", "Manila", "Hong Kong", "Mumbai"]),
            remote=rng.choice(["no", "hybrid", "yes"]),
            contract=rng.choice(["Permanent", "Contract", "Permanent"]),
            salary_min=rng.choice([60000, 84000, 96000, 120000, 144000]),
            salary_max=rng.choice([96000, 120000, 144000, 180000, 216000]),
            currency=rng.choice(["MYR", "SGD", "USD", "IDR"]),
            must=must, nice=nice,
            responsibilities=[
                "Own regional delivery and stakeholder management.",
                "Drive performance against agreed quarterly objectives.",
                "Collaborate with product, operations and commercial teams.",
            ],
        ))
    return rows


# --------------------------------------------------------------------------- #
# PDF writer
# --------------------------------------------------------------------------- #
def _write_pdf(path: Path, title: str, lines: list[str], header: str,
               footer: str, page_break_after: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=9.2, leading=12.2, spaceAfter=2)
    head = ParagraphStyle("head", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=12.5, leading=15, spaceBefore=5, spaceAfter=4)
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=20 * mm, bottomMargin=18 * mm,
                            title=title, author="Second Talent (synthetic demo)")

    def on_page(canv, doc_):
        canv.saveState()
        canv.setFont("Helvetica", 7.5)
        canv.setFillColorRGB(0.35, 0.35, 0.35)
        canv.drawString(18 * mm, A4[1] - 12 * mm, header)
        canv.drawString(18 * mm, 10 * mm, footer.format(page=doc_.page))
        canv.setStrokeColorRGB(0.8, 0.8, 0.8)
        canv.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
        canv.restoreState()

    story = []
    for line in lines:
        s = line.strip()
        if not s:
            story.append(Spacer(1, 3))
        elif s.isupper() and len(s) < 60 and not s.startswith("-"):
            story.append(Paragraph(s.replace("&", "&amp;"), head))
        elif s.startswith("- "):
            story.append(Paragraph("• " + s[2:].replace("&", "&amp;"), body))
        elif s.startswith("  - "):
            story.append(Paragraph("&nbsp;&nbsp;&nbsp;– " + s[4:].replace("&", "&amp;"), body))
        else:
            story.append(Paragraph(s.replace("&", "&amp;"), body))
        if page_break_after is not None and len(story) == page_break_after:
            story.append(PageBreak())
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def _merge_pdfs(out: Path, parts: list[Path]) -> None:
    import pymupdf

    with pymupdf.open() as merged:
        for part in parts:
            with pymupdf.open(part) as src:
                merged.insert_pdf(src)
        merged.save(out)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_synthetic_source(dest: Path, seed: int = 42, n_pool: int = 100,
                           n_jd: int = 50) -> Path:
    marker = dest / ".complete"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "CV").mkdir(exist_ok=True)
    (dest / "JD+HiredCV").mkdir(exist_ok=True)
    rng = random.Random(seed)

    # ---------------------------------------------------------------- CVs
    cv_rows = []
    for i in range(1, n_pool + 1):
        doc_id = f"CV-{i:03d}"
        profile = _profile(doc_id, rng)
        lines = _cv_lines(doc_id, profile, rng)
        _write_pdf(dest / "CV" / f"{doc_id}.pdf", doc_id, lines,
                   header="Anonymised candidate profile · personal information removed · Second Talent",
                   footer=f"{doc_id} · page {{page}}")
        cv_rows.append(dict(
            Ref=doc_id, **{
                "Current title": profile["title"],
                "Current company": profile["company"],
                "Country": profile["country"],
                "Role family": profile["family"],
                "Skills": ", ".join(profile["skills"]),
                "Work history": "; ".join(
                    f"{r['title']} at {r['company']} ({r['start']}–{r['end']})"
                    for r in _work_history(rng, profile, doc_id)),
                "Education": "Bachelor's Degree",
                "Latest degree / university": "Bachelor's Degree, " + rng.choice(UNIVERSITIES),
            }
        ))

    # ---------------------------------------------------------------- GT JDs + hired CVs
    for jd in HIRED_PROFILES:
        jd_part = dest / "JD+HiredCV" / f"_{jd['jd_id']}_jd.pdf"
        cv_part = dest / "JD+HiredCV" / f"_{jd['jd_id']}_cv.pdf"
        _write_pdf(jd_part, jd["jd_title"], _jd_lines(jd),
                   header=f"{jd['jd_id']} · JOB DESCRIPTION · {jd['family']}",
                   footer=f"{jd['jd_id']} · page {{page}}")
        profile = dict(title=jd["title"], company=jd["company"], family=jd["family"],
                       country=jd["country"], years=jd["years"], seniority=jd["seniority"],
                       skills=jd["skills"])
        cv_lines = ["CV OF THE HIRED CANDIDATE", f"Hired for: {jd['jd_title']}", ""] + \
                   _cv_lines("", profile, rng)
        _write_pdf(cv_part, jd["title"], cv_lines,
                   header=f"{jd['jd_id']} · CANDIDATE CV · HIRED",
                   footer=f"{jd['jd_id']} · page {{page}}")
        _merge_pdfs(dest / "JD+HiredCV" / f"{jd['jd_id']}.pdf", [jd_part, cv_part])
        jd_part.unlink()
        cv_part.unlink()

    # ---------------------------------------------------------------- unlabelled JDs
    jd_rows = []
    for jd in _unlabelled_jds(seed, n_jd):
        desc_lines = [f"JOB TITLE: {jd['position']}", f"ROLE FAMILY: {jd['family']}",
                      f"LOCATION: {jd['city']}, {jd['country']}", f"REMOTE: {jd['remote']}",
                      "", "ABOUT THE ROLE",
                      f"We are looking for a {jd['position']} to join our {jd['family']} team.",
                      "", "KEY RESPONSIBILITIES"]
        desc_lines += [f"- {r}" for r in jd["responsibilities"]]
        desc_lines += ["", "REQUIREMENTS (MUST-HAVE)"] + [f"- {s}" for s in jd["must"]]
        desc_lines += ["", "NICE TO HAVE"] + [f"- {s}" for s in jd["nice"]]
        jd_rows.append({
            "#": jd["num"], "Position": jd["position"], "Role family": jd["family"],
            "Country": jd["country"], "City": jd["city"], "Remote": jd["remote"],
            "Contract type": jd["contract"], "Salary min": jd["salary_min"],
            "Salary max": jd["salary_max"], "Currency": jd["currency"],
            "Job description": "\n".join(desc_lines),
        })

    excel_path = dest / "Manatal Examples – JD & CV (anonymised).xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(jd_rows).to_excel(writer, sheet_name="JD Examples", index=False)
        pd.DataFrame(cv_rows).to_excel(writer, sheet_name="CV Examples", index=False)

    # ---------------------------------------------------------------- brief PDF
    _write_pdf(dest / "AI_CV_Matching_Project_Plan.pdf", "AI CV Matching Project Plan", [
        "PROJECT OBJECTIVE",
        "Build a reproducible baseline that ranks candidate CVs for a job description and "
        "returns a short, recruiter-readable reason for each match.",
        "",
        "RECRUITER PRIORITIES",
        "Hard-skill overlap and years of relevant experience are highest priority. Seniority and "
        "salary fit are high priority. Industry background, job stability and education are medium. "
        "Location, time zone, notice period and language are filters.",
        "",
        "PRIVACY",
        "All documents are anonymised. Use a local LLM only. Do not upload data or attempt to "
        "re-identify candidates.",
        "",
        "SCOPE OF THIS BASELINE",
        "Keyword extraction with a local Ollama model, keyword merging and clustering, a shared "
        "autoencoder representation, top-10 retrieval, Recall@10 evaluation and an error analysis. "
        "Salary, notice period and hard location/language filters are out of scope and must be "
        "reported as limitations.",
        "",
        "NOTE",
        "This PDF is part of the synthetic demo corpus generated because DATA/ was absent in the "
        "execution environment. It is not the original assignment brief.",
    ], header="Anonymised example · client and personal information removed · Second Talent",
        footer="page {page}")

    marker.write_text("synthetic demo source complete\n", encoding="utf-8")
    return dest
