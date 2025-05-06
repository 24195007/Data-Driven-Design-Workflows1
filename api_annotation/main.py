import os, base64, time, csv, json, re
from pathlib import Path
from openai import OpenAI
from openai._exceptions import RateLimitError, APIError, OpenAIError

# Configuration
client = OpenAI(api_key="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")  # 替换为你自己的 Key
IMAGES_DIR = Path("../crawl/converted_jpg")
OUTPUT_CSV = Path("annotations.csv")
MODEL = "gpt-4o"

# Initialize CSV header
if not OUTPUT_CSV.exists():
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "filename", "Safety", "Belonging", "Naturalness", "Shared", "Privacy", "Convenience", "concept_text"
        ])

# Load processed images
processed = set()
with OUTPUT_CSV.open("r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        if row:
            processed.add(row[0])

# Filter images to process
images_to_process = [
    img for img in os.listdir(IMAGES_DIR)
    if img.lower().endswith((".jpg", ".png", ".webp")) and img not in processed
]

def annotate_worker(img_name):
    img_path = IMAGES_DIR / img_name
    try:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please evaluate the architectural image on the following six dimensions with integer scores from 1 to 5: "
                            "Safety, Belonging, Naturalness, Shared, Privacy, and Convenience. "
                            "Then, generate a short concept phrase inspired by Greek architectural context (e.g., 'Shared rooftop at sunset').\n"
                            "Return the result strictly in the following JSON format:\n"
                            "{\n"
                            "  \"Safety\": integer (1-5),\n"
                            "  \"Belonging\": integer (1-5),\n"
                            "  \"Naturalness\": integer (1-5),\n"
                            "  \"Shared\": integer (1-5),\n"
                            "  \"Privacy\": integer (1-5),\n"
                            "  \"Convenience\": integer (1-5),\n"
                            "  \"concept\": \"short concept phrase\"\n"
                            "}"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    }
                ]
            }
        ]

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=512
                )
                text = resp.choices[0].message.content.strip()

                # Extract JSON from markdown block if wrapped
                match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)

                data = json.loads(text)
                scores = [
                    data["Safety"], data["Belonging"], data["Naturalness"],
                    data["Shared"], data["Privacy"], data["Convenience"]
                ]
                concept = data["concept"]
                return [img_name, *scores, concept]
            except (RateLimitError, APIError, OpenAIError, json.JSONDecodeError, KeyError) as e:
                print(f"[{img_name}] Attempt {attempt + 1} failed: {e}")
                time.sleep(2 * (attempt + 1))

        print(f"[{img_name}] Annotation failed. Skipping.")
        return None

    except Exception as e:
        print(f"[{img_name}] Failed to load image: {e}")
        return None

if __name__ == "__main__":
    for img_name in images_to_process:
        result = annotate_worker(img_name)

        # Write each result to CSV
        if result:
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(result)
                print(f"[Saved] {result[0]} completed")