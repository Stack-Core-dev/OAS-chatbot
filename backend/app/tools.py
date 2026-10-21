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

    return _format_hits(retrieve(query, k=4, sources=POLICY_SOURCES))


@tool
def search_digital_daak(query: str) -> str:
    """Search the Digital Daak manual, visual layout reference, and support contacts.

    Use for questions about Digital Daak features, screens, workflows,
    dispatch, receiving, tracking, troubleshooting, or support-team contacts.
    Pass focused keywords.
    """
    from app.rag import retrieve

    return _format_hits(retrieve(query, k=4, sources=DIGITAL_DAAK_SOURCES))


@tool
def lookup_contact(name: str = "", department: str = "", extension: str = "") -> str:
    """Look up AWT employee contact details in the local directory.

    Use for any person lookup: finding a name, phone extension, mobile number,
    designation, department, or floor. Also use when the user asks who a person
    is or which person belongs to an extension or department. Provide at least
    one of name, department, or extension. Partial names, misspelled names, and
    department names are accepted. For role questions, put the role in name
    and the department in department.
    """
    from app.contacts import lookup_contacts

    matches = lookup_contacts(name=name, department=department, extension=extension)
    if not matches:
        return "No matching contact found."
    return "\n".join(
        f"Name: {contact['name']} | Designation: {contact['designation']} | "
        f"Department: {contact['department']} | Floor: {contact['floor']} | "
        f"Extension: {contact['extension']} | Mobile: {contact['mobile']}"
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

    return _format_hits(retrieve(query, k=4))


TOOLS = [
    get_current_datetime,
    search_policies,
    search_digital_daak,
    lookup_contact,
    generate_letter,
    search_documents,
]
