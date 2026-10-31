import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle

def build_pdf():
    pdf_path = os.path.join(os.getcwd(), "LOWKEY_Project_Plan.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1E1E2E'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#5C5F77'),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E66F5'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#313244'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#202020'),
        backColor=colors.HexColor('#F5F5F7'),
        borderPadding=6,
        spaceAfter=8
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("LOWKEY &mdash; Project Plan &amp; Roadmap", title_style))
    story.append(Paragraph("<b>Core Idea:</b> Discover songs you didn't know you'd love. LOWKEY recommends lesser-known tracks that match your taste while helping you explore beyond your usual music bubble.", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E66F5'), spaceAfter=12))

    # Decisions Made So Far
    story.append(Paragraph("1. Decisions Made So Far", h1_style))
    story.append(Paragraph("&bull; <b>Project Name:</b> LOWKEY.", bullet_style))
    story.append(Paragraph("&bull; <b>Core Experience:</b> User enters a song and receives recommendations that preserve its vibe while introducing unexpected discoveries.", bullet_style))
    story.append(Paragraph("&bull; <b>Two Recommendation Signals:</b><br/>"
                           "&nbsp;&nbsp;&nbsp;&nbsp;1. <i>Content-based similarity:</i> Find similar songs using audio features or embeddings.<br/>"
                           "&nbsp;&nbsp;&nbsp;&nbsp;2. <i>Collaborative filtering:</i> Find songs listened to by people who also listened to the input song.", bullet_style))
    story.append(Paragraph("&bull; <b>Discovery Layer:</b> Prioritize lesser-known tracks without sacrificing musical relevance.", bullet_style))
    story.append(Paragraph("&bull; <b>Data Strategy:</b> Prefer one static dataset supporting both audio similarity and listening-history analysis, avoiding continuous syncing or dependence on live APIs.", bullet_style))

    # Proposed Workflow
    story.append(Paragraph("2. Proposed Workflow", h1_style))
    story.append(Paragraph("&bull; <b>Step 1:</b> User enters a song.", bullet_style))
    story.append(Paragraph("&bull; <b>Step 2:</b> Retrieve candidate songs using musical similarity and listener overlap.", bullet_style))
    story.append(Paragraph("&bull; <b>Step 3:</b> Rank candidates using relevance, popularity/underheard signals, and diversity.", bullet_style))
    story.append(Paragraph("&bull; <b>Step 4:</b> Display recommendations with explanations for why each song was selected.", bullet_style))
    story.append(Paragraph("&bull; <b>Step 5:</b> Let users refine recommendations using natural language feedback.", bullet_style))

    # Dataset Being Evaluated
    story.append(Paragraph("3. Dataset Being Evaluated", h1_style))
    story.append(Paragraph("&bull; <b>#nowplaying-RS:</b> A potential single-dataset solution containing historical listening events and audio features.", bullet_style))
    story.append(Paragraph("&bull; <b>Verification Required:</b> Verify track-ID-to-song-name mapping, available audio features, coverage, and licensing.", bullet_style))
    story.append(Paragraph("&bull; <b>Contingency:</b> If it cannot support both recommendation signals adequately, evaluate alternative datasets.", bullet_style))

    # Still To Be Decided
    story.append(Paragraph("4. Still To Be Decided", h1_style))
    story.append(Paragraph("&bull; Final dataset and track-metadata mapping.", bullet_style))
    story.append(Paragraph("&bull; Whether to use traditional audio features, pretrained audio embeddings, or both.", bullet_style))
    story.append(Paragraph("&bull; Collaborative-filtering method and how to combine it with content similarity.", bullet_style))
    story.append(Paragraph("&bull; Metric to measure whether a recommendation is genuinely less mainstream.", bullet_style))
    story.append(Paragraph("&bull; Final ranking formula and fallback for songs missing from the dataset.", bullet_style))
    story.append(Paragraph("&bull; Frontend design and the scope of the three-day MVP.", bullet_style))

    # Team Responsibilities & MVP Goal
    story.append(Paragraph("5. Team Structure &amp; MVP Goal", h1_style))
    story.append(Paragraph("&bull; <b>AI / Full-stack:</b> Data pipeline, recommendation integration, backend APIs, and frontend.", bullet_style))
    story.append(Paragraph("&bull; <b>ML:</b> Dataset analysis, feature-based similarity, collaborative filtering, ranking experiments, and evaluation.", bullet_style))
    story.append(Paragraph("&bull; <b>MVP Goal:</b> Build a working song-discovery system that recommends musically relevant but less-obvious tracks using both audio characteristics and historical listening patterns.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CCD0DA'), spaceAfter=10))

    # Today's To-Do List
    story.append(Paragraph("Today's Task Specification: Audio Similarity Retrieval using HNSW (ANN)", h1_style))
    story.append(Paragraph("<b>Target Dataset:</b> <code>backend/data/Train-Test Splits/Sequence/train_SEQ.csv</code>", body_style))
    story.append(Paragraph("<b>11 Audio Feature Dimensions:</b> danceability, energy, valence, tempo, loudness, acousticness, instrumentalness, liveness, speechiness, key, mode.", body_style))
    story.append(Paragraph("<b>Input Query Payload (Cleaned &amp; Normalized):</b>", body_style))

    payload_text = (
        '{\n'
        '  "danceability": 0.9802, "energy": 0.9793, "valence": 0.5625,\n'
        '  "bpm": 116.69, "loudness_db": -20.82, "acousticness": 0.0071,\n'
        '  "instrumentalness": 0.1396, "liveness": 0.0, "speechiness": 0.0,\n'
        '  "key_int": 0, "mode": 1\n'
        '}\n'
        'Field Mapping: bpm -> tempo, loudness_db -> loudness, key_int -> key'
    )
    story.append(Paragraph(payload_text.replace('\n', '<br/>'), code_style))

    story.append(Paragraph("<b>Core Objectives:</b>", body_style))
    story.append(Paragraph("&bull; 1. Load and deduplicate <code>train_SEQ.csv</code> by <code>track_id</code>.", bullet_style))
    story.append(Paragraph("&bull; 2. Normalize all 11 features to [0, 1] using <code>MinMaxScaler</code>.", bullet_style))
    story.append(Paragraph("&bull; 3. Build a vector graph using HNSW (<code>faiss.IndexHNSWFlat</code> or <code>hnswlib</code>).", bullet_style))
    story.append(Paragraph("&bull; 4. Perform Approximate Nearest Neighbor (ANN) search to return Top 5 most similar tracks with track IDs and Cosine Similarity scores.", bullet_style))

    doc.build(story)
    print(f"PDF generated successfully at: {pdf_path}")

if __name__ == "__main__":
    build_pdf()

