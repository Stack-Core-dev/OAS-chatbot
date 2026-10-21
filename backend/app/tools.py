from datetime import datetime

from langchain_core.tools import tool

POLICY_SOURCES = (
    "PNP25.pdf",
    "CorporateProfile.pdf",
    "MD.pdf",
)

DIGITAL_DAAK_SOURCES = (
    "DigitalDAAKManual.pdf",
    "Digital_Daak_Visual_Layout_Reference_for_OAS_Chatbot.pdf",
    "Digital_Daak_Support_Team_Contacts.pdf",
)


def _format_hits(hits) -> str:
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(
        f"[{source}, page {page}] {text.strip()[:700]}"
        for text, source, page, _ in hits
    )


@tool
def get_current_datetime() -> str:
    """Get the current system date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


@tool
def search_policies(query: str) -> str:
    """Search AWT/Askari organization policies and corporate documents.

    Use for questions about HR, leave, finance, procurement, administration,
    regulations, or organizational structure. Pass focused keywords.
    """
    from app.rag import retrieve

    return _format_hits(retrieve(query, sources=POLICY_SOURCES))


@tool
def search_digital_daak(query: str) -> str:
    """Search the Digital Daak manual, visual layout reference, and support contacts.

    Use for questions about Digital Daak features, screens, workflows,
    dispatch, receiving, tracking, troubleshooting, or support-team contacts.
    Pass focused keywords.
    """
    from app.rag import retrieve

    return _format_hits(retrieve(query, sources=DIGITAL_DAAK_SOURCES))


@tool
def lookup_contact(name: str = "", department: str = "", extension: str = "") -> str:
    """Look up AWT employee contact details in the local directory.

    Use for any person lookup: finding a name, contact number, extension,
    designation, department, or floor. Also use when the user asks who a person
    is or which person belongs to an extension or department. Provide at least
    one of name, department, or extension. Partial names, misspelled names, and
    department names are accepted. For role questions, put the role in name
    and the department in department.

    Directory abbreviations (use these to translate full names before calling):
    Business units: AWT = Army Welfare Trust; BU = Business Unit;
    RE = Real Estate; AE = Askari Enterprises; AGL = Askari Guards Limited;
    AGICO = Askari General Insurance Company; ALACO = Askari Life Assurance
    Company; ADHL = Askari Development & Holdings; AWT IL / AWTIL = AWT
    Investments Ltd; AWSM = Army Welfare Sugar Mills; ATW / ATT = Askari Travel
    Wing / Askari Travel & Tours; AAS = Askari Aviation Services;
    FSS = Fauji Security Services; MALP = MAL Pakistan Ltd;
    ASK = Askari; ask mob = Askari Mobil; KCI = Karachi investment/real-estate
    unit label; seads = Seeds (directory spelling).
    Locations: LHR = Lahore; RWP = Rawalpindi; FSD = Faisalabad; BDN = Badin.
    Posts: ED = Executive Director; MD = Managing Director;
    COO = Chief Operating Officer; CFO = Chief Financial Officer;
    CTO = Chief Technology Officer; DHR = Director Human Resources;
    DMA / M&A = Director Monitoring & Audit; GH = Group Head;
    GM = General Manager; HoD = Head of Department; SM = Senior Manager;
    SAM = Senior Assistant Manager; DM / Dy Manager = Deputy Manager;
    AM = Assistant Manager; SMF = Senior Manager Finance; Mgr = Manager;
    PA = Personal Assistant; SO = Staff Officer; Secy = Secretary;
    Sectt = Secretariat; EDS = ED Secretariat; Supdt = Superintendent;
    Offr = Officer; Ops = Operations; Proc = Procurement;
    QA = Quality Assurance; HRBP / HR BP = HR Business Partner;
    Fin = Finance; Avn = Aviation; Svcs / Svc = Services;
    IF&T / IF & T = Industries, Farms & Tourism;
    Avn & Fin Svcs / A&FS / A/FS = Aviation & Financial Services;
    ADM&HR = Administration & Human Resources;
    RMD = Risk Management Department; IA = Internal Audit;
    MIS = Management Information Systems; SW = Software; DC = Data Centre;
    DC-MIS = Data Centre MIS; DC-Exchange = Data Centre Exchange;
    MT = Motor Transport; BD = Business Development;
    R&S = Recruitment & Selection; HCM = Human Capital Management;
    CT = Corporate Tax; SAP PM = Plant Maintenance;
    SAP MM = Materials Management; SAP FICO = Finance & Controlling;
    SAP ABAP = Advanced Business Application Programming;
    SAP BASIS = Business Application Software Integrated Solution.
    Other shorthand: Snr = Senior; Asst = Assistant; Dy = Deputy;
    Aud = Audit; Maint = Maintenance; Coord = Coordination;
    Deptt = Department; Con Room = Conference Room; ENGR = Engineering;
    HR Ops = HR Operations; Assoc = Associate; Tel Op = Telephone Operator;
    Tel Tech = Telephone Technician; Clk = Clerk; Hav = Havildar;
    Lnk = Lance Naik; N/Sub = Naib Subedar; Sub = Subedar;
    Hon = Honorary; Capt = Captain; Maj = Major; Lt Col = Lieutenant Colonel;
    Col = Colonel; Brig = Brigadier; Maj Gen = Major General;
    Lt Gen = Lieutenant General; Retd = Retired; HI(M) = Hilal-i-Imtiaz
    (Military); SI(M) = Sitara-i-Imtiaz (Military).
    """
    from app.contacts import lookup_contacts

    matches = lookup_contacts(name=name, department=department, extension=extension)
    if not matches:
        return "No matching contact found."
    return "\n".join(
        f"Name: {contact['name']} | Designation: {contact['designation']} | "
        f"Department: {contact['department']} | Floor: {contact['floor']} | "
        f"Contact: {contact['contact']}"
        for contact in matches
    )


@tool
async def generate_letter(
    subject: str,
    letter_type: str = "general",
    recipient_name: str = "",
    key_points: str = "",
    recipient_designation: str = "",
    recipient_department: str = "",
    tone: str = "formal",
    sender_name: str = "",
    sender_designation: str = "",
) -> str:
    """Draft an AWT business letter, certificate, NOC, or similar document.

    Use for requests to draft, write, or create correspondence. Only the subject
    is required; use any known details and leave unknown details blank. Do not
    ask for every field. The tool inserts clear bracketed placeholders for
    missing facts and tells the user to replace them. Wording stays simple and
    the body has at least two paragraphs. Do not invent missing facts.
    """
    from app.letter_agent import generate_letter as create_letter

    return await create_letter(
        subject=subject,
        letter_type=letter_type,
        recipient_name=recipient_name,
        key_points=key_points,
        recipient_designation=recipient_designation,
        recipient_department=recipient_department,
        tone=tone,
        sender_name=sender_name,
        sender_designation=sender_designation,
    )


@tool
def search_documents(query: str) -> str:
    """Search the entire AWT document knowledge base.

    Use only when the question spans multiple areas or neither the policies
    nor Digital Daak search tool applies. Pass focused keywords, not the whole
    question.
    """
    from app.rag import retrieve

    return _format_hits(retrieve(query))


TOOLS = [
    get_current_datetime,
    search_policies,
    search_digital_daak,
    lookup_contact,
    generate_letter,
    search_documents,
]
