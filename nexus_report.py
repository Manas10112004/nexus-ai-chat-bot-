import streamlit as st
from fpdf import FPDF
import os


class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Nexus AI - Session Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_pdf(history, session_id):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)

    pdf.cell(0, 10, f"Session ID: {session_id}", ln=True)
    pdf.ln(5)

    for msg in history:
        role = msg["role"].upper()
        content = msg["content"]

        # Clean text (handle special chars slightly)
        content = content.encode('latin-1', 'replace').decode('latin-1')

        # Role Header
        pdf.set_font("Arial", 'B', 10)
        pdf.set_text_color(0, 50, 150) if role == "USER" else pdf.set_text_color(0, 100, 50)
        pdf.cell(0, 6, f"[{role}]", ln=True)

        # Content
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, content)
        pdf.ln(3)

    # Check for latest chart
    if os.path.exists("temp_chart.png"):
        pdf.add_page()
        pdf.cell(0, 10, "Attached Analysis Chart:", ln=True)
        pdf.image("temp_chart.png", x=10, y=30, w=180)

    output_filename = f"report_{session_id}.pdf"
    pdf.output(output_filename)
    return output_filename