import pandas as pd
import re
from langchain_core.documents import Document


def extract_price(text):
    """Try to find a price mention like '1599/-', 'Rs 1500', '@1599', '1565/-rs' in review text."""
    if not isinstance(text, str):
        return None
    patterns = [
        r'(?:rs\.?|inr|₹)\s?(\d{3,6})',
        r'(\d{3,6})\s?/-',
        r'@\s?(\d{3,6})',
        r'bought.{0,15}?(\d{3,6})',
        r'purchased.{0,15}?(\d{3,6})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            price = int(match.group(1))
            if 200 <= price <= 50000:  # sanity range for earphones/headsets
                return price
    return None


def dataconverter():
    data = pd.read_csv(r"C:\Users\AAKAR\OneDrive\Documents\ECOMMERCE_RECOMMENDATION\flipkart_product_review.csv")
    data = data[["product_title", "rating", "review"]]

    # Aggregate average rating and detected prices per product
    product_stats = {}
    for _, row in data.iterrows():
        name = row["product_title"]
        if name not in product_stats:
            product_stats[name] = {"ratings": [], "prices": []}
        try:
            product_stats[name]["ratings"].append(float(row["rating"]))
        except (ValueError, TypeError):
            pass
        price = extract_price(row["review"])
        if price:
            product_stats[name]["prices"].append(price)

    summary = {}
    for name, stats in product_stats.items():
        avg_rating = round(sum(stats["ratings"]) / len(stats["ratings"]), 1) if stats["ratings"] else None
        if stats["prices"]:
            # Use the most frequently mentioned price as the representative single value
            price_counts = {}
            for p in stats["prices"]:
                price_counts[p] = price_counts.get(p, 0) + 1
            most_common_price = max(price_counts.items(), key=lambda x: x[1])[0]
            price_text = f"₹{most_common_price}"
        else:
            price_text = "Not available"
        summary[name] = {"avg_rating": avg_rating, "price": price_text}

    docs = []
    for _, row in data.iterrows():
        name = row["product_title"]
        metadata = {
            "product_name": name,
            "avg_rating": summary[name]["avg_rating"],
            "price": summary[name]["price"],
        }
        doc = Document(page_content=row["review"], metadata=metadata)
        docs.append(doc)
    return docs