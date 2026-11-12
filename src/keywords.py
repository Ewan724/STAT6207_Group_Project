import json
import re
import unicodedata
from pathlib import Path

CURATED_ALIASES = [
    ("js","JavaScript"),("java script","JavaScript"),("javascript","JavaScript"),
    ("ts","TypeScript"),("typescript","TypeScript"),
    ("gcp","Google Cloud Platform"),("google cloud","Google Cloud Platform"),("google cloud platform","Google Cloud Platform"),
    ("aws","AWS"),("amazon web services","AWS"),
    ("azure","Microsoft Azure"),("microsoft azure","Microsoft Azure"),
    ("k8s","Kubernetes"),("kube","Kubernetes"),("kubernetes","Kubernetes"),
    ("postgres","PostgreSQL"),("postgresql","PostgreSQL"),("psql","PostgreSQL"),
    ("mysql","MySQL"),("my sql","MySQL"),
    ("mssql","SQL Server"),("sql server","SQL Server"),("microsoft sql server","SQL Server"),
    ("ml","Machine Learning"),("machine learning","Machine Learning"),
    ("ai","Artificial Intelligence"),("artificial intelligence","Artificial Intelligence"),
    ("nlp","Natural Language Processing"),("natural language processing","Natural Language Processing"),
    ("llm","Large Language Model"),("large language model","Large Language Model"),("large language models","Large Language Model"),
    ("rag","Retrieval-Augmented Generation"),("retrieval augmented generation","Retrieval-Augmented Generation"),("retrieval-augmented generation","Retrieval-Augmented Generation"),
    ("ci/cd","CI/CD"),("cicd","CI/CD"),("continuous integration continuous deployment","CI/CD"),
    ("devops","DevOps"),("dev ops","DevOps"),
    ("node.js","Node.js"),("nodejs","Node.js"),("node","Node.js"),
    ("react.js","React"),("reactjs","React"),("react","React"),
    ("vue.js","Vue.js"),("vuejs","Vue.js"),("vue","Vue.js"),
    ("next.js","Next.js"),("nextjs","Next.js"),
    ("c#","C#"),("c sharp","C#"),("csharp","C#"),
    ("c++","C++"),("cpp","C++"),
    (".net","DotNet"),("dotnet",".NET"),
    ("power bi","Power BI"),("powerbi","Power BI"),
    ("tableau","Tableau"),
    ("excel","Excel"),("microsoft excel","Excel"),("ms excel","Excel"),
    ("ifrs","IFRS"),("international financial reporting standards","IFRS"),
    ("gaap","US GAAP"),("us gaap","US GAAP"),
    ("fp&a","FP&A"),("fpa","FP&A"),("financial planning and analysis","FP&A"),
    ("erp","ERP"),("enterprise resource planning","ERP"),
    ("wms","Warehouse Management System"),("warehouse management system","WMS"),
    ("oms","Order Management System"),("order management system","OMS"),
    ("crm","Customer Relationship Management"),("customer relationship management","CRM"),
    ("seo","SEO"),("search engine optimisation","SEO"),("search engine optimization","SEO"),
    ("sem","SEM"),("search engine marketing","SEM"),
    ("gtm","Go-to-Market"),("go-to-market","Go-to-Market"),("go to market","Go-to-Market"),
    ("b2b","B2B"),("business to business","B2B"),
    ("b2c","B2C"),("business to consumer","B2C"),
    ("p&l","P&L"),("pnl","P&L"),("profit and loss","P&L"),
    ("kpi","KPI"),("kpis","KPI"),("key performance indicator","KPI"),
    ("sla","SLA"),("slas","SLA"),("service level agreement","SLA"),
    ("api","API"),("apis","API"),("application programming interface","API"),
    ("sdk","SDK"),("sdks","SDK"),
    ("etl","ETL"),("extract transform load","ETL"),
    ("elt","ELT"),("extract load transform","ELT"),
    ("dbt","dbt"),("data build tool","dbt"),
    ("sql","SQL"),("structured query language","SQL"),
    ("nosql","NoSQL"),("no sql","NoSQL"),
    ("ui","UI"),("user interface","UI"),
    ("ux","UX"),("user experience","UX"),
    ("figma","Figma"),
    ("jira","JIRA"),
    ("agile","Agile"),("agile methodology","Agile"),
    ("scrum","Scrum"),
    ("spark","Apache Spark"),("apache spark","Apache Spark"),
    ("airflow","Apache Airflow"),("apache airflow","Apache Airflow"),
]

PROTECTED = {"kubernetes","postgresql","javascript","typescript","node.js","next.js","react","vue.js","c#","c++","dotnet",".net"}

CONFUSABLE_BLOCKLIST = [
    ({"java"},{"javascript"}),
    ({"c"},{"c++","c#"}),
    ({"c++"},{"c#"}),
    ({"react"},{"react native"}),
    ({"sql"},{"nosql"}),
    ({"mysql"},{"sql server"}),
    ({"azure"},{"aws","gcp"}),
    ({"aws"},{"gcp"}),
    ({"power bi"},{"tableau"}),
    ({"wms"},{"oms"}),
    ({"ui"},{"ux"}),
]


def normalize_keyword_str(s):
    if s is None: return None
    t = unicodedata.normalize("NFKC", str(s))
    t = t.strip().lower()
    t = t.replace("\u00ad","")
    # normalize some chars
    t = re.sub(r"[‘’“”]", "'", t)
    t = re.sub(r"\s+", " ", t)
    # don't blanket strip dots/slashes for known
    return t.strip(" \t\n\r-•,.;:()[]{}!")


def build_alias_map():
    m = {}
    for raw, canon in CURATED_ALIASES:
        m[normalize_keyword_str(raw)] = canon  # store canonical as proper form
    return m


def apply_alias(s, alias_map):
    n = normalize_keyword_str(s)
    if n in alias_map:
        return alias_map[n]
    # simple plural
    if n.endswith("s") and len(n)>4 and n not in PROTECTED:
        base = n[:-1]
        if base in alias_map:
            return alias_map[base]
    return s.strip()  # keep original form if not aliased? better return title-like form
