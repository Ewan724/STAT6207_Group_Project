"""Curated keyword alias dictionary and over-merge guardrails.

The dictionary is authoritative. Variants on the left map to one canonical
term on the right. Distinct technologies deliberately never share a canonical
term (Java vs JavaScript, C vs C++, SQL vs NoSQL, and so on).
"""
from __future__ import annotations

import re
import unicodedata


def _norm_key(value: str) -> str:
    s = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9+#./ -]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# canonical: [variants]
CURATED_ALIASES: dict[str, list[str]] = {
    "JavaScript": ["js", "java script", "javascript", "java-script"],
    "TypeScript": ["ts", "typescript", "type script"],
    "Google Cloud Platform": ["gcp", "google cloud", "google cloud platform"],
    "AWS": ["aws", "amazon web services"],
    "Microsoft Azure": ["azure", "microsoft azure", "ms azure"],
    "Kubernetes": ["k8s", "kube", "kubernetes"],
    "PostgreSQL": ["postgres", "postgresql", "psql"],
    "MySQL": ["mysql", "my sql"],
    "SQL Server": ["mssql", "sql server", "microsoft sql server", "ms sql"],
    "Machine Learning": ["ml", "machine learning"],
    "Artificial Intelligence": ["ai", "artificial intelligence"],
    "Natural Language Processing": ["nlp", "natural language processing"],
    "Large Language Model": ["llm", "llms", "large language model", "large language models"],
    "Retrieval-Augmented Generation": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration continuous deployment", "continuous integration"],
    "DevOps": ["devops", "dev ops"],
    "Node.js": ["node.js", "nodejs", "node"],
    "React": ["react.js", "reactjs", "react"],
    "Vue.js": ["vue.js", "vuejs", "vue"],
    "Next.js": ["next.js", "nextjs"],
    "C#": ["c#", "c sharp", "csharp"],
    "C++": ["c++", "cpp"],
    ".NET": [".net", "dotnet", "asp.net"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "microsoft excel", "ms excel"],
    "IFRS": ["ifrs", "international financial reporting standards"],
    "US GAAP": ["gaap", "us gaap"],
    "FP&A": ["fp&a", "fpa", "financial planning and analysis"],
    "ERP": ["erp", "enterprise resource planning"],
    "WMS": ["wms", "warehouse management system", "warehouse management systems"],
    "OMS": ["oms", "order management system", "order management systems"],
    "CRM": ["crm", "customer relationship management"],
    "SEO": ["seo", "search engine optimisation", "search engine optimization"],
    "SEM": ["sem", "search engine marketing"],
    "Go-to-Market": ["gtm", "go-to-market", "go to market"],
    "B2B": ["b2b", "business to business"],
    "B2C": ["b2c", "business to consumer"],
    "P&L": ["p&l", "pnl", "profit and loss"],
    "KPI": ["kpi", "kpis", "key performance indicator", "key performance indicators"],
    "SLA": ["sla", "slas", "service level agreement", "service level agreements"],
    "API": ["api", "apis", "application programming interface"],
    "SDK": ["sdk", "sdks"],
    "ETL": ["etl", "extract transform load"],
    "ELT": ["elt", "extract load transform"],
    "dbt": ["dbt", "data build tool"],
    "SQL": ["sql", "structured query language"],
    "NoSQL": ["nosql", "no sql"],
    "UI": ["ui", "user interface"],
    "UX": ["ux", "user experience"],
    "Figma": ["figma"],
    "JIRA": ["jira"],
    "Agile": ["agile", "agile methodology"],
    "Scrum": ["scrum"],
    "Golang": ["golang", "go language"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "tf"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Apache Spark": ["spark", "apache spark"],
    "Apache Airflow": ["airflow", "apache airflow"],
    "Snowflake": ["snowflake"],
    "BigQuery": ["bigquery", "big query"],
    "Docker": ["docker"],
    "Terraform": ["terraform"],
    "Jenkins": ["jenkins"],
    "Ansible": ["ansible"],
    "Git": ["git"],
    "Linux": ["linux"],
    "REST API": ["rest api", "restful api", "rest"],
    "Microservices": ["microservices", "microservice"],
    "Kafka": ["kafka", "apache kafka"],
    "RabbitMQ": ["rabbitmq"],
    "Elasticsearch": ["elasticsearch", "elastic search"],
    "MongoDB": ["mongodb", "mongo db"],
    "Redis": ["redis"],
    "Salesforce": ["salesforce"],
    "HubSpot": ["hubspot"],
    "Google Analytics": ["google analytics", "ga4"],
    "Google Ads": ["google ads", "adwords", "google adwords"],
    "Meta Ads": ["meta ads", "facebook ads"],
    "Adobe Photoshop": ["photoshop", "adobe photoshop"],
    "Adobe Illustrator": ["illustrator", "adobe illustrator"],
    "Adobe XD": ["adobe xd", "xd"],
    "Sketch": ["sketch"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Java": ["java"],
    "Spring Boot": ["spring boot"],
    "PHP": ["php"],
    "Laravel": ["laravel"],
    "Django": ["django"],
    "Angular": ["angular", "angularjs"],
    "SAP": ["sap"],
    "Workday": ["workday"],
    "SAP SuccessFactors": ["successfactors", "sap successfactors"],
    "Product Management": ["product management"],
    "Product Strategy": ["product strategy"],
    "Roadmap": ["roadmap", "product roadmap"],
    "User Research": ["user research"],
    "A/B Testing": ["a/b testing", "ab testing", "split testing"],
    "Design System": ["design system", "design systems"],
    "Prototyping": ["prototyping", "prototype"],
    "Wireframing": ["wireframing", "wireframe"],
    "Usability Testing": ["usability testing"],
    "Interaction Design": ["interaction design"],
    "Accessibility": ["accessibility", "a11y"],
    "Financial Reporting": ["financial reporting"],
    "Financial Analysis": ["financial analysis"],
    "Budgeting": ["budgeting", "budget"],
    "Forecasting": ["forecasting", "forecast"],
    "Cash Flow Management": ["cash flow management", "cash flow"],
    "Internal Controls": ["internal controls", "internal control"],
    "Manufacturing Costing": ["manufacturing costing", "costing"],
    "Taxation": ["taxation", "tax planning", "tax"],
    "Audit": ["audit", "auditing"],
    "Talent Acquisition": ["talent acquisition"],
    "Recruitment": ["recruitment", "recruiting"],
    "Payroll": ["payroll"],
    "Employee Relations": ["employee relations"],
    "Performance Management": ["performance management"],
    "Compensation and Benefits": ["compensation and benefits", "compensation & benefits", "compensation"],
    "Onboarding": ["onboarding"],
    "HRIS": ["hris", "human resource information system"],
    "Field Service": ["field service"],
    "Preventive Maintenance": ["preventive maintenance", "preventative maintenance"],
    "Troubleshooting": ["troubleshooting"],
    "Semiconductor Equipment": ["semiconductor equipment", "semiconductor"],
    "Cleanroom": ["cleanroom", "clean room"],
    "PLC": ["plc", "programmable logic controller"],
    "Electrical Systems": ["electrical systems", "electrical"],
    "Technical Documentation": ["technical documentation"],
    "Stakeholder Management": ["stakeholder management", "stakeholder communication"],
    "Key Account Management": ["key account management", "account management"],
    "Business Development": ["business development"],
    "Lead Generation": ["lead generation"],
    "Sales Pipeline": ["sales pipeline", "pipeline management"],
    "Negotiation": ["negotiation", "negotiating"],
    "Demand Generation": ["demand generation"],
    "Content Marketing": ["content marketing"],
    "Email Marketing": ["email marketing"],
    "Marketing Automation": ["marketing automation"],
    "Brand Management": ["brand management", "branding"],
    "Campaign Management": ["campaign management", "campaigns"],
    "Market Research": ["market research"],
    "Data Warehousing": ["data warehousing", "data warehouse"],
    "Statistics": ["statistics", "statistical analysis"],
    "Deep Learning": ["deep learning"],
    "Data Pipeline": ["data pipeline", "data pipelines"],
    "Prompt Engineering": ["prompt engineering"],
    "Vector Database": ["vector database", "vector databases", "vector search"],
}

ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, variants in CURATED_ALIASES.items():
    ALIAS_TO_CANONICAL[_norm_key(canonical)] = canonical
    for v in variants:
        ALIAS_TO_CANONICAL[_norm_key(v)] = canonical


# Pairs that must never be in the same cluster (canonical forms).
BLOCKED_PAIRS = [
    ("Java", "JavaScript"),
    ("C", "C++"),
    ("C#", "C++"),
    ("React", "React Native"),
    ("SQL", "NoSQL"),
    ("MySQL", "SQL Server"),
    ("Microsoft Azure", "AWS"),
    ("Google Cloud Platform", "AWS"),
    ("Power BI", "Tableau"),
    ("WMS", "OMS"),
    ("UI", "UX"),
    ("Angular", "AngularJS"),
    ("TensorFlow", "PyTorch"),
    ("Docker", "Kubernetes"),
    ("ETL", "ELT"),
]


def canonical_for(term: str) -> str:
    return ALIAS_TO_CANONICAL.get(_norm_key(term), str(term).strip())


def conflicts(a: str, b: str) -> bool:
    ca, cb = _norm_key(canonical_for(a)), _norm_key(canonical_for(b))
    for x, y in BLOCKED_PAIRS:
        nx, ny = _norm_key(x), _norm_key(y)
        if (ca == nx and cb == ny) or (ca == ny and cb == nx):
            return True
    return False
