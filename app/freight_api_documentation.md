# 🚚 Freight Estimation API Documentation

**Last updated:** 2025-04-08 17:58:57

## 🔗 Base URL
```
http://localhost:8000
```

---

## 📤 Endpoints

### 1. `POST /estimate`
Estimate both **CWT-based** and **Area-based** freight cost for a single product.

#### Request Body (JSON)
```json
{
  "quantity": 1500,
  "conversion_code": "Carpet Tile_1CPT_SQFT",
  "site": "SPT"
}
```

#### Response Example
```json
{
  "estimated_cwt_cost": 156.98,
  "freight_class_cwt": "L5C",
  "rate_cwt": 25.37,
  "discount_cwt": 1,
  "estimated_area_cost": 130.57,
  "freight_class_area": "L5C",
  "rate_area": 0.7834,
  "discount_area": 1,
  "commodity_group": "1CPT",
  "uom": "SQFT"
}
```

---

### 2. `POST /batch`
Estimate freight for multiple records via an uploaded file.

#### Accepted File Types
- `.csv`, `.xls`, `.xlsx`

#### Required Columns
- `SITEID`
- `QUANTITY`
- `CONVERSIONCODE`

#### Optional Columns
- `SUPPLIERID`, `SUPPLIERNAME`, `PARTNUMBER`, `COMMODITYGROUP`, `COMMODITYDESCRIPTION`, `UOM`

#### Response Preview
```json
{
  "filename": "sample_batch_input.csv",
  "rows_processed": 3,
  "preview": [
    {
      "SITEID": "SPT",
      "QUANTITY": 1500,
      ...
      "estimated_cwt_cost": 156.98,
      "estimated_area_cost": 130.57
    }
  ],
  "download_url": "/download/freight_dual_results_20250408_123456.csv"
}
```

---

### 3. `GET /download/{filename}`
Download results after batch processing.

#### Example
```
GET /download/freight_dual_results_20250408_123456.csv
```

---

## 🗂 Supporting Files

- `freight_rates_updated.csv`
- `conversion_table_with_code.csv`

---

## ⚙️ Running Locally with Logs

```bash
uvicorn freight_api:app --reload --log-level debug
```

To log to file:
```bash
uvicorn freight_api:app --reload > logs.txt 2>&1
```

---

## 🧪 Testing Options

### Swagger UI
```
http://localhost:8000/docs
```

### Using `curl`
```bash
curl -X 'POST' 'http://127.0.0.1:8000/batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_batch_input.csv;type=text/csv'
```

---
