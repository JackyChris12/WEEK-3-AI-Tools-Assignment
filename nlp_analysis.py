import spacy
import bz2
import re

def get_nlp_model():
    """Loads and returns the spaCy model."""
    # This function can be cached by the caller (e.g., Streamlit)
    return spacy.load("en_core_web_sm")

def extract_review_text(line):
    """Extract the review text from the FastText format"""
    # FastText format: __label__X review_text
    match = re.search(r'__label__\d+\s+(.*)', line)
    if match:
        return match.group(1)
    return line.strip()

def rule_based_sentiment(text):
    """Simple rule-based sentiment analysis"""
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'best', 
                     'awesome', 'wonderful', 'fantastic', 'good', 'loved', 'nice',
                     'happy', 'satisfied', 'recommend', 'quality', 'beautiful']
    
    # Negative keywords
    negative_words = ['bad', 'terrible', 'worst', 'hate', 'poor', 'awful', 
                     'disappointed', 'useless', 'horrible', 'waste', 'broken',
                     'defective', 'cheap', 'not good', 'not recommend', 'refund']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

def process_reviews(file_path, num_reviews=10):
    """Process Amazon reviews from compressed file"""
    print(f"Processing reviews from: {file_path}\n")
    print("=" * 80)
    
    nlp = get_nlp_model() # Load model for standalone script execution
    with bz2.open(file_path, 'rt', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= num_reviews:
                break
            
            # Extract review text
            review_text = extract_review_text(line)
            
            # Skip if review is too short
            if len(review_text) < 20:
                continue
            
            # Process with spaCy
            doc = nlp(review_text[:500])  # Limit to first 500 chars for efficiency
            
            # Extract entities (focusing on PRODUCT, ORG, and PERSON which may indicate brands)
            entities = []
            for ent in doc.ents:
                if ent.label_ in ['PRODUCT', 'ORG', 'PERSON', 'GPE']:
                    entities.append((ent.text, ent.label_))
            
            # Perform sentiment analysis
            sentiment = rule_based_sentiment(review_text)
            
            # Display results
            print(f"\nReview #{i + 1}:")
            print("-" * 80)
            print(f"Text: {review_text[:200]}..." if len(review_text) > 200 else f"Text: {review_text}")
            print(f"\nSentiment: {sentiment}")
            print(f"\nExtracted Entities:")
            if entities:
                for entity, label in entities:
                    print(f"  - {entity} ({label})")
            else:
                print("  - No entities found")
            print("=" * 80)

def main():
    """Main function to run the NLP analysis"""
    print("\n" + "=" * 80)
    print("NLP ANALYSIS WITH spaCy - Amazon Product Reviews")
    print("=" * 80 + "\n")
    
    # Process training data
    train_file = "archive (1)/train.ft.txt.bz2"
    
    print("Analyzing sample reviews...\n")
    process_reviews(train_file, num_reviews=10)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print("- Named Entity Recognition (NER) performed using spaCy")
    print("- Extracted entities include: PRODUCT, ORG (Organizations/Brands), PERSON, GPE")
    print("- Sentiment analysis using rule-based approach with keyword matching")
    print("- Sentiments classified as: Positive, Negative, or Neutral")

if __name__ == "__main__":
    main()
