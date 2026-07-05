import sys
sys.path.append('.')

from ingest import ingestor
from vector_store import vector_store
from logger import logger

# Test 1: Check Qdrant connection
try:
    stats = vector_store.get_stats()
    print(f"✅ Qdrant connected: {stats}")
except Exception as e:
    print(f"❌ Qdrant connection failed: {e}")
    sys.exit(1)

# Test 2: Ingest a text file (create a simple PDF)
# Since you may not have a simple PDF, let's create one
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Test Knowledge Base", ln=True)
pdf.cell(200, 10, txt="This is a test document for the Knowledge Engine.", ln=True)
pdf.cell(200, 10, txt="AI and machine learning are transforming how we work.", ln=True)
pdf.output("data/documents/test.pdf")

print("✅ Created test.pdf")

# Ingest
result = ingestor.ingest_pdf("data/documents/test.pdf")
print(f"Ingestion result: {result}")
print(f"Success: {result.success}")
print(f"Chunks: {result.chunks_created}")
if result.error:
    print(f"Error: {result.error}")