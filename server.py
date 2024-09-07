from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import os
import pandas as pd
from playwright.sync_api import sync_playwright
from textblob import TextBlob  # NLP library for sentiment analysis
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import secrets  # For generating a secure random key

app = Flask(__name__)

# Set directories for storing screenshots and comments
screenshot_dir = "static/images"
os.makedirs(screenshot_dir, exist_ok=True)

# Set the secret key for session management
app.secret_key = secrets.token_hex(16)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_url():
    instagram_url = request.form.get('instagram_url')

    if not instagram_url:
        flash("Please provide an Instagram post URL.")
        return redirect(url_for('index'))

    # Call the function to process the Instagram post
    try:
        extracted_text, sentiment, screenshot_path = take_screenshot_and_extract_text(instagram_url)
        filename = screenshot_path.split('/')[-1]
        return render_template('result.html', image_path=filename, text=extracted_text, sentiment=sentiment)
    except Exception as e:
        flash(f"Error processing the Instagram post: {e}")
        return redirect(url_for('index'))

def take_screenshot_and_extract_text(url: str):
    """
    Scrape an Instagram post, take a screenshot, extract comments, and apply NLP.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)  # Launch browser in non-headless mode
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # Go to the Instagram post URL
        page.goto(url)

        try:
            # Wait for the post to load
            page.wait_for_selector("article", timeout=10000)

            # Extract post ID from URL
            post_id = url.split('/')[-2]

            # Step 1: Capture screenshot of the post (add delay to fully load comments)
            page.wait_for_timeout(5000)  # Allow some time for the comments to load
            screenshot_path = os.path.join(screenshot_dir, f"post_{post_id}.png")
            page.screenshot(path=screenshot_path)

            # Step 2: Extract text from the screenshot using OCR
            extracted_text = extract_text_from_image(screenshot_path)

            if extracted_text:
                # Step 3: Apply NLP to the extracted text (Sentiment Analysis using TextBlob)
                sentiment = analyze_text(extracted_text)
                return extracted_text, sentiment, screenshot_path
            else:
                raise Exception("No text found in the screenshot!")

        except Exception as e:
            raise e

        finally:
            browser.close()

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR with preprocessing.
    """
    img = Image.open(image_path)

    # Convert to grayscale for better OCR accuracy
    img = ImageOps.grayscale(img)

    # Increase contrast to make the text more readable
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Increase contrast

    # Run OCR with custom configurations
    custom_config = r'--oem 3 --psm 6'  # PSM 6 is suitable for text blocks
    text = pytesseract.image_to_string(img, config=custom_config)

    return text.strip()

def analyze_text(text: str) -> float:
    """
    Analyze the text using basic sentiment analysis and return sentiment polarity.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Sentiment polarity: -1 (negative) to 1 (positive)

@app.route('/download_image/<filename>')
def download_image(filename):
    """
    Download the saved image.
    """
    return send_file(os.path.join(screenshot_dir, filename), as_attachment=True)

@app.route('/download_excel')
def download_excel():
    """
    Save extracted text to an Excel file.
    """
    text = request.args.get('text')  # Fetch text from query params
    if not text:
        flash("No text available for generating Excel.")
        return redirect(url_for('index'))

    excel_path = os.path.join(screenshot_dir, 'extracted_data.xlsx')
    df = pd.DataFrame({'Extracted Text': [text]})
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

@app.route('/download_pdf')
def download_pdf():
    """
    Save extracted text to a PDF file using reportlab.
    """
    text = request.args.get('text')  # Fetch text from query params
    if not text:
        flash("No text available for generating PDF.")
        return redirect(url_for('index'))

    try:
        # Ensure directory exists
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        pdf_path = os.path.join(screenshot_dir, 'extracted_data_reportlab.pdf')

        # Create a PDF document using reportlab
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)

        # Define margin and line height
        text_object = c.beginText(40, 750)
        text_object.setFont("Helvetica", 12)
        text_object.textLines(text)

        # Write text to PDF
        c.drawText(text_object)
        c.showPage()
        c.save()

        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        flash(f"Error generating PDF: {e}")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
