# AI Framework for Comparing Different Data Formats

## Overview
This project compares two versions of the same information and automatically detects inconsistencies using an LLM. It identifies missing, incorrect, conflicting, and hallucinated information, then generates a structured evaluation report.

## Use Case
**Audio Transcript vs Summary**

## Features
- Detects missing information
- Identifies incorrect values
- Finds conflicting information
- Detects hallucinated (extra) information
- Generates structured evaluation reports

## Test Cases
The project includes:
- Perfect matches
- Missing information
- Incorrect values
- Ambiguous cases
- Sensitive information

## Sample Output
```json
{
  "missing_information": [],
  "incorrect_information": [],
  "conflicting_information": [],
  "hallucinated_information": [],
  "overall_score": 98,
  "status": "Match"
}
```


