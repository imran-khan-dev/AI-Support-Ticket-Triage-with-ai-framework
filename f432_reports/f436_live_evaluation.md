# F4.3.6 Complete Live Answer Quality Evaluation

- **Experiment ID:** 9eea7ddb-42f8-4462-b462-357b71909f64
- **Dataset:** support-kb-answer-quality-eval (v1.0.0)
- **Tenant:** eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee
- **Provider:** Groq (openai/gpt-oss-20b)
- **Generation mode:** Real LLM
- **Evaluation mode:** Deterministic + Genuine Ragas
- **Ragas available:** False
- **Overall status:** completed

## Generation Results Summary

| Implementation | Generated | Failed | Provider/Model |
|----------------|-----------|--------|----------------|
| Custom | 1 | 6 | Groq (openai/gpt-oss-20b) |
| Framework | 0 | 7 | Groq (openai/gpt-oss-20b) |

## Retrieval Isolation Evidence

| Sample | Custom Backend | Framework Backend | Custom Chunks | Framework Chunks |
|--------|---------------|-------------------|---------------|------------------|
| aq1 | custom-retrieval-stack | llamaindex | C1, C2, C10 | C1, C2, C10 |
| aq2 | custom-retrieval-stack | llamaindex | C3, C6, C4, C1, C2 | C3, C4, C6 |
| aq3 | custom-retrieval-stack | llamaindex | C5, C6, C9, C1, C2 | C5, C6 |
| aq4 | custom-retrieval-stack | llamaindex | C10, C2, C4, C8, C3 | C10, C4, C3, C8 |
| aq5 | custom-retrieval-stack | llamaindex | C8, C7 | C8, C7 |
| aq6 | custom-retrieval-stack | llamaindex | C9, C2, C5 | C9, C2, C5 |
| aq7 | custom-retrieval-stack | llamaindex | C5, C6, C8, C10, C2 |  |

## Per-Sample Evidence

### aq1: How do I reset my password?
- **Reference:** Go to account settings, click Forgot password, check your email for a reset link, and follow the link to create a new password.

#### Custom
- **Status:** success
- **Backend:** custom-retrieval-stack
- **Chunks:** C1, C2, C10
- **Answer:** To reset your password, go to **Account Settings** and click **“Forgot password.”**  
A reset link will be sent to your email address. Open that email, click the link, and follow the instructions to create a new password and regain access to your account.  
You can also change your password directly
- **Error:** None

#### Framework
- **Status:** provider_failure
- **Backend:** llamaindex
- **Chunks:** C1, C2, C10
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

### aq2: How long do billing refunds take to process?
- **Reference:** Billing refunds are processed within 5 business days for eligible charges and are issued to the original payment method.

#### Custom
- **Status:** provider_failure
- **Backend:** custom-retrieval-stack
- **Chunks:** C3, C6, C4, C1, C2
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Backend:** llamaindex
- **Chunks:** C3, C4, C6
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

### aq3: How long does shipping take and do I get tracking information?
- **Reference:** Orders ship within 24 hours and tracking information is provided via email.

#### Custom
- **Status:** provider_failure
- **Backend:** custom-retrieval-stack
- **Chunks:** C5, C6, C9, C1, C2
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Backend:** llamaindex
- **Chunks:** C5, C6
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

### aq4: Can I get a refund for a purchase from 6 months ago?
- **Reference:** The knowledge base does not state a time limit for requesting refunds. Billing refunds are processed within 5 business days for eligible charges. Contact support if your refund has not arrived after 10 business days.

#### Custom
- **Status:** provider_failure
- **Backend:** custom-retrieval-stack
- **Chunks:** C10, C2, C4, C8, C3
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Backend:** llamaindex
- **Chunks:** C10, C4, C3, C8
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

### aq5: How do I check my API rate limit usage?
- **Reference:** You can view your rate limit usage in the developer dashboard.

#### Custom
- **Status:** provider_failure
- **Backend:** custom-retrieval-stack
- **Chunks:** C8, C7
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Backend:** llamaindex
- **Chunks:** C8, C7
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

### aq6: How do I update my email notification preferences?
- **Reference:** Account settings let you update your profile, email address, and notification preferences.

#### Custom
- **Status:** provider_failure
- **Backend:** custom-retrieval-stack
- **Chunks:** C9, C2, C5
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Backend:** llamaindex
- **Chunks:** C9, C2, C5
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

### aq7: What is the company's stock price?
- **Reference:** I don't have information about the company's stock price.

#### Custom
- **Status:** provider_failure
- **Backend:** custom-retrieval-stack
- **Chunks:** C5, C6, C8, C10, C2
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Backend:** llamaindex
- **Chunks:** 
- **Answer:** None
- **Error:** Provider service is temporarily unavailable.

## Deterministic Metrics (Diagnostic)

| Metric | Custom Mean | Framework Mean | Custom Count | Framework Count |
|--------|-------------|----------------|--------------|-----------------|
| answer_correctness (custom) | 0.1063 | | 7 | |
| answer_relevancy (custom) | 0.0387 | | 7 | |
| faithfulness (custom) | 0.1224 | | 7 | |
| context_precision (custom) | 0.3571 | | 7 | |
| context_recall (custom) | 0.8571 | | 7 | |
| answer_correctness (framework) | 0.0000 | | 7 | |
| answer_relevancy (framework) | 0.0000 | | 7 | |
| faithfulness (framework) | 0.0000 | | 7 | |
| context_precision (framework) | 0.4524 | | 7 | |
| context_recall (framework) | 0.8571 | | 7 | |

## Genuine Ragas Metrics

**Not available.** Ragas evaluation did not complete.
Deterministic metrics are used instead.

No NaN scores were treated as valid metrics.