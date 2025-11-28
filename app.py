import io
import os
import google.generativeai as genai
from groq import Groq
from datetime import datetime
from PyPDF2 import PdfReader


from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from dotenv import load_dotenv
#from openai import OpenAI  # kept for future BYO-key or paid mode, not used now
from docx import Document
#from weasyprint import HTML  # HTML -> PDF
# from weasyprint import HTML
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path
from dotenv import load_dotenv
from flask_session import Session


# --- ENV & APP SETUP ---------------------------------------------------------
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)
#print("Loaded .env from:", dotenv_path)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY   = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
secret = os.getenv("FLASK_SECRET_KEY")
if not secret:
    raise RuntimeError("FLASK_SECRET_KEY is required")
app.secret_key = secret


app.config["SESSION_TYPE"] = "filesystem"   # store session data in local files
app.config["SESSION_FILE_DIR"] = "./.flask_session"  # any writable folder
app.config["SESSION_PERMANENT"] = False
Session(app)

# If/when you want OpenAI again, uncomment these lines and set env keys:
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MODEL = "auto"



# --- PROVIDER ADAPTERS (Gemini, Groq) ----------------------------------------
def gemini_generate(prompt: str, system: str = "") -> str:
    if not GEMINI_KEY:
        raise RuntimeError("Gemini key missing")
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.5-pro")
    parts = []
    if system:
        parts.append({"text": system})
    parts.append({"text": prompt})
    res = model.generate_content(parts, safety_settings=None)
    return (res.text or "").strip()

def groq_generate(prompt: str, system: str = "") -> str:
    if not GROQ_KEY:
        raise RuntimeError("Groq key missing")
    client = Groq(api_key=GROQ_KEY)
    model_name = "llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant" for speed
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    chat = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.6,
        max_tokens=800,
    )
    return (chat.choices[0].message.content or "").strip()


# --- HELPERS -----------------------------------------------------------------
def extract_company(job_desc: str) -> str | None:
    for line in job_desc.splitlines():
        if line.strip().lower().startswith("company:"):
            return line.split(":", 1)[1].strip()
    return None

def build_formatted_header(
    full_name: str,
    email: str,
    linkedin: str,
    github: str,
    date_pretty: str,
    company: str | None,
    portfolio: str = ""
) -> str:
    lines = []
    if full_name:
        lines.append(full_name)
    if email:
        lines.append(f"Email: {email}")
    if github:
        lines.append(f"GitHub: {github}")
    if linkedin:
        lines.append(f"LinkedIn: {linkedin}")
    if portfolio:
        lines.append(f"Portfolio: {portfolio}")

    # date line
    if date_pretty:
        if lines:
            lines.append("")  # space before date if any contact lines exist
        lines.append(date_pretty)
        lines.append("")

    # greeting
    lines.append(f"Dear {company} Hiring Team," if company else "Dear Hiring Team,")
    lines.append("")  # blank line before body

    return "\n".join(lines)


def format_preview(
    full_name: str,
    email: str,
    linkedin: str,
    github: str,
    date_pretty: str,
    company: str | None,
    body: str,
    closing: str = "Best regards",
    portfolio: str = ""
) -> str:
    header = build_formatted_header(full_name, email, linkedin, github, date_pretty, company, portfolio)
    paras = [p.strip() for p in body.strip().split("\n\n") if p.strip()]
    body_clean = "\n\n".join(paras)
    footer = f"\n\n{closing},"
    if full_name:
        footer += f"\n{full_name}"

    return header + body_clean + footer

def extract_text_from_upload(file_storage) -> str:
    """Return text from an uploaded .txt, .md, .docx, or .pdf file."""
    if not file_storage or not file_storage.filename:
        return ""
    name = file_storage.filename.lower()
    data = file_storage.read()

    if name.endswith(".txt") or name.endswith(".md"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    elif name.endswith(".docx"):
        try:
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    elif name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            chunks = []
            for page in reader.pages:
                text = page.extract_text() or ""
                chunks.append(text)
            return "\n".join(chunks)
        except Exception:
            return ""
    else:
        return ""


def build_prompt_body_only(resume_text: str, job_desc: str, word_limit: int | None, extra_instructions: str = "") -> str:
    parts = [
        f"Using ONLY the following resume content:\n{resume_text}\n\n",
        f"Write the BODY of a professional cover letter for this job:\n{job_desc}\n\n",
        "IMPORTANT RULES:\n",
        "- Output ONLY the body paragraphs. Do NOT include a header, contact lines, date, greeting line, or signature block.\n",
        "- Do NOT add tools/skills that are not explicitly present in the resume.\n",
        f"- Keep it under {word_limit or 300} words.\n",
    ]
    extra = (extra_instructions or "").strip()
    if extra:
        parts += [
            "\nADDITIONAL USER PREFERENCES (follow carefully, but still obey the rules above):\n",
            extra + "\n",
        ]
    return "".join(parts)


# --- ROUTES ------------------------------------------------------------------
@app.get("/")
def index():
    preview = session.get("preview", "")
    jd = session.get("job_description", "")
    resume = session.get("resume_text", "")
    header_email = ""
    header_linkedin = ""
    header_github = ""
    header_portfolio = ""
    word_limit = session.get("word_limit", 300)
    model = session.get("model", DEFAULT_MODEL)  # just a display field now

    today_iso = datetime.today().strftime("%Y-%m-%d")
    letter_date_iso = session.get("letter_date_iso", today_iso)

    extra_instructions = session.get("extra_instructions", "")

    return render_template(
        "index.html",
        preview=preview,
        job_description=jd,
        resume_text=resume,
        header_email=header_email,
        header_linkedin=header_linkedin,
        header_github=header_github,
        header_portfolio=header_portfolio,
        word_limit=word_limit,
        model=model,
        today=datetime.today().strftime("%B %d, %Y"),
        letter_date_iso=letter_date_iso,
        extra_instructions=extra_instructions,
    )


@app.post("/generate")
def generate():
    # 0) Read textareas
    jd = request.form.get("job_description", "").strip()
    resume = request.form.get("resume_text", "").strip()

    # 1) Optional file uploads override text
    jd_file = request.files.get("jd_file")
    resume_file = request.files.get("resume_file")
    jd_from_file = extract_text_from_upload(jd_file) if jd_file and jd_file.filename else ""
    resume_from_file = extract_text_from_upload(resume_file) if resume_file and resume_file.filename else ""
    if jd_from_file:
        jd = jd_from_file
    if resume_from_file:
        resume = resume_from_file

    # 2) Contact + options
    header_email = (request.form.get("header_email") or "").strip()
    header_linkedin = (request.form.get("header_linkedin") or "").strip()
    header_github = (request.form.get("header_github") or "").strip()
    header_portfolio = (request.form.get("header_portfolio") or "").strip()
    
    extra_instructions = request.form.get("extra_instructions", "").strip()
    full_name = (request.form.get("full_name") or "").strip()

    word_limit_raw = request.form.get("word_limit", "300").strip()
    provider_pref = (request.form.get("model", DEFAULT_MODEL) or DEFAULT_MODEL).lower()
    try:
        word_limit = int(word_limit_raw) if word_limit_raw else None
    except ValueError:
        word_limit = 300

    # 3) Date
    letter_date_iso = request.form.get("letter_date") or datetime.today().strftime("%Y-%m-%d")
    try:
        letter_date_pretty = datetime.strptime(letter_date_iso, "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        letter_date_iso = datetime.today().strftime("%Y-%m-%d")
        letter_date_pretty = datetime.today().strftime("%B %d, %Y")

    # 4) Validate inputs
    if not jd or not resume:
        flash("Please provide both a job description and resume (paste or upload).", "error")
        session.update({
            "preview": session.get("preview", ""),
            "job_description": jd,
            "resume_text": resume,
            "word_limit": word_limit,
            "model": provider_pref,
            "letter_date_iso": letter_date_iso,
            "extra_instructions": extra_instructions,
        })
        return redirect(url_for("index"))

    # 5) Greeting company (optional)
    company_name = extract_company(jd)

    # 6) Prompt (BODY ONLY)
    prompt = build_prompt_body_only(resume, jd, word_limit, extra_instructions)

    # 7) Provider Selection Logic
    # We define the 'try_provider' helper and the 'order' list BEFORE the loop
    def try_provider(name: str) -> str:
        system_prompt = "You write concise, accurate cover letter bodies strictly from the user's resume text."
        if name == "gemini":
            return gemini_generate(prompt, system_prompt)
        elif name == "groq":
            return groq_generate(prompt, system_prompt)
        else:
            raise RuntimeError("Unknown provider")

    if provider_pref == "auto":
        order = ["gemini", "groq"]
    elif provider_pref == "gemini":
        order = ["gemini", "groq"]
    elif provider_pref == "groq":
        order = ["groq", "gemini"]
    else:
        order = ["gemini", "groq"]

    body = ""
    provider_used = None
    last_err = None

    # 8) The Loop (with your print debugging added)
    for idx, p in enumerate(order):
        try:
            body = try_provider(p)
            provider_used = p
            if idx > 0:
                flash(f"{order[0].title()} limit/error encountered; used {p.title()} instead.", "info")
            break
        except Exception as e:
            # --- THIS IS THE DEBUG LINE ---
            print(f"❌ ERROR with {p}: {e}")
            # ------------------------------
            last_err = e
            continue

    if not body:
        flash(f"All providers failed: {last_err}", "error")
        body = session.get("preview", "")

    # 9) Compose formatted preview
    if body:
        preview = format_preview(
            full_name=full_name,
            email=header_email,
            linkedin=header_linkedin,
            github=header_github,
            portfolio=header_portfolio,
            date_pretty=letter_date_pretty,
            company=company_name,
            body=body,
            closing="Best regards",
        )
        if provider_used and provider_used == provider_pref:
            flash(f"Generated with {provider_used.title()}.", "success")
    else:
        preview = session.get("preview", "")

    # 10) Persist fields
    session.update({
        "preview": preview,
        "job_description": jd,
        "resume_text": resume,
        "word_limit": word_limit,
        "model": provider_pref,
        "letter_date_iso": letter_date_iso,
        "extra_instructions": extra_instructions,
    })

    return redirect(url_for("index"))

    

    # ---- Provider selection & fallback logic --------------------------------
    def try_provider(name: str) -> str:
        system_prompt = "You write concise, accurate cover letter bodies strictly from the user's resume text."
        if name == "gemini":
            return gemini_generate(prompt, system_prompt)
        elif name == "groq":
            return groq_generate(prompt, system_prompt)
        else:
            raise RuntimeError("Unknown provider")

    if provider_pref == "auto":
        order = ["gemini", "groq"]
    elif provider_pref == "gemini":
        order = ["gemini", "groq"]   # fallback to groq if gemini fails
    elif provider_pref == "groq":
        order = ["groq", "gemini"]   # fallback to gemini if groq fails
    else:
        order = ["gemini", "groq"]

    body = ""
    provider_used = None
    last_err = None

    for idx, p in enumerate(order):
        try:
            body = try_provider(p)
            provider_used = p
            if idx > 0:
                flash(f"{order[0].title()} limit/error encountered; used {p.title()} instead.", "info")
            break
        except Exception as e:
            last_err = e
            continue

    if not body:
        flash(f"All providers failed: {last_err}", "error")
        body = session.get("preview", "")

    full_name = (request.form.get("full_name") or "").strip()



    # 7) Compose formatted preview
    if body:
        preview = format_preview(
            full_name=full_name,
            email=header_email,
            linkedin=header_linkedin,
            github=header_github,
            portfolio=header_portfolio,
            date_pretty=letter_date_pretty,
            company=company_name,
            body=body,
            closing="Best regards",
        )
        if provider_used:
            flash(f"Generated with {provider_used.title()}.", "success")
    else:
        preview = session.get("preview", "")

    # 8) Persist fields for sticky form + preview
    session.update({
        "preview": preview,
        "job_description": jd,
        "resume_text": resume,
        "word_limit": word_limit,
        "model": provider_pref,
        "letter_date_iso": letter_date_iso,
        "extra_instructions": extra_instructions,
    })

    return redirect(url_for("index"))



@app.post("/download-pdf")
def download_pdf():
    content = session.get("preview", "").strip()
    if not content:
        flash("No generated cover letter to download.")
        return redirect(url_for("index"))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    left = 1 * inch
    right = width - 1 * inch
    top = height - 1 * inch

    # basic styles
    NAME_FONT = "Helvetica-Bold"
    BODY_FONT = "Helvetica"
    NAME_SIZE = 14
    BODY_SIZE = 11
    LEADING = 16  # line spacing
    PARA_SPACING = 10

    y = top

    def draw_line(text, font=BODY_FONT, size=BODY_SIZE):
        nonlocal y
        c.setFont(font, size)
        c.drawString(left, y, text)
        y -= LEADING
        if y < 1 * inch:
            c.showPage()
            y = top

    def draw_wrapped_paragraph(text, font=BODY_FONT, size=BODY_SIZE, max_width=right-left):
        nonlocal y
        from reportlab.pdfbase import pdfmetrics
        words = text.split()
        line = ""
        c.setFont(font, size)
        for w in words:
            trial = (line + " " + w).strip()
            if pdfmetrics.stringWidth(trial, font, size) > max_width:
                c.drawString(left, y, line)
                y -= LEADING
                if y < 1 * inch:
                    c.showPage()
                    y = top
                line = w
            else:
                line = trial
        if line:
            c.drawString(left, y, line)
            y -= LEADING
        y -= PARA_SPACING
        if y < 1 * inch:
            c.showPage()
            y = top

    lines = content.splitlines()

    # Bold only if the first line is a name (i.e., not empty and not starting with "Email:" or "Dear ")
    i = 0
    if lines:
        first = lines[0].strip()
        if first and not first.startswith(("Email:", "GitHub:", "LinkedIn:", "Portfolio:", "Dear ")):
            draw_line(first, font=NAME_FONT, size=NAME_SIZE)
            i = 1

    while i < len(lines) and not lines[i].startswith("Dear "):
        line = lines[i]
        if line.strip() == "":
            y -= PARA_SPACING
        else:
            draw_line(line)
        i += 1

    # Greeting line
    if i < len(lines) and lines[i].startswith("Dear "):
        y -= PARA_SPACING // 2
        draw_line(lines[i])
        i += 1
        y -= PARA_SPACING // 2

    # Body paragraphs until a closing marker
    paras: list[str] = []
    tail: list[str] = []
    in_body = True
    while i < len(lines):
        line = lines[i]
        if line.strip().endswith(",") and ("regards" in line.lower() or "sincerely" in line.lower()):
            in_body = False
        if in_body:
            paras.append(line)
        else:
            tail.append(line)
        i += 1

    body_text = "\n".join(paras)
    body_paras = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    for p in body_paras:
        draw_wrapped_paragraph(p)

    # Closing/signature
    for t in tail:
        if t.strip():
            draw_line(t)
        else:
            y -= PARA_SPACING

    c.showPage()
    c.save()
    buf.seek(0)

    fname = f"Cover_Letter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=fname)


@app.post("/clear")
def clear():
    session.clear()
    return redirect(url_for("index"))


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5050, debug=True)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))  # Railway provides this automatically
    app.run(host="0.0.0.0", port=port, debug=False)

