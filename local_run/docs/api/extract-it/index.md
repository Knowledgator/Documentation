# Comprehend-it API

The Comprehend-it API provides a zero-shot text classification service using a natural language inference (NLI) model. It processes input text and classifies it into specified labels, returning a probability score for each label.

---

## Endpoint

`POST https://comprehend-it.p.rapidapi.com/predictions/ml-zero-nli-model`

---

## Headers

| Name              | Type   | Description                                 |
|-------------------|--------|---------------------------------------------|
| `X-RapidAPI-Key`  | String | Your RapidAPI key.                          |
| `X-RapidAPI-Host` | String | `comprehend-it.p.rapidapi.com`              |
| `Content-Type`    | String | `application/json`                          |

---

## Request Body

```json
{
  "labels": ["label1", "label2", "label3"],
  "text": "Your input text here.",
  "threshold": 0.5
}
```

- `labels` (required): List of categories for classification.
- `text` (required): The text to classify.
- `threshold` (optional): Minimum score threshold for label inclusion.

---

## Response

```json
{
  "outputs": {
    "label1": 0.85,
    "label2": 0.10,
    "label3": 0.05
  }
}
```

Each label is associated with a probability score indicating the model's confidence.

---

## Error Codes

| Status Code | Description                               |
|-------------|-------------------------------------------|
| 200         | OK – Request successful.                  |
| 400         | Bad Request – Invalid or incomplete input.|
| 401         | Unauthorized – Invalid API key.           |
| 404         | Not Found – Invalid endpoint or payload.  |
| 413         | Payload Too Large – Exceeds 30 KB limit.  |
| 500         | Internal Server Error – Server-side issue.|

---

## Example Usage

```python
import requests

url = "https://comprehend-it.p.rapidapi.com/predictions/ml-zero-nli-model"
headers = {
    "Content-Type": "application/json",
    "X-RapidAPI-Key": "YOUR_API_KEY",
    "X-RapidAPI-Host": "comprehend-it.p.rapidapi.com"
}
payload = {
    "labels": ["positive", "negative", "neutral"],
    "text": "I love this product!",
    "threshold": 0.5
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

---

## Additional Information

- **Input Size Limit**: The maximum payload size is 30 KB.
- **Label Flexibility**: You can specify any set of labels relevant to your classification task.
- **Use Cases**: Sentiment analysis, topic classification, intent detection, etc.

For more details, refer to the [Comprehend-it API Documentation](https://docs.knowledgator.com/docs/api-reference/comprehend-it-api).
