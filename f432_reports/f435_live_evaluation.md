# F4.3.5 Complete Live Answer Quality Evaluation

- **Experiment ID:** 8673e862-b8a3-41b3-8b1f-0e3def55e60f
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
| Custom | 2 | 5 | Groq (openai/gpt-oss-20b) |
| Framework | 2 | 5 | Groq (openai/gpt-oss-20b) |

## Retrieval Isolation Evidence

| Sample | Custom Backend | Framework Backend | Custom Chunks | Framework Chunks |
|--------|---------------|-------------------|---------------|------------------|
| aq1 | custom-retrieval-stack | llamaindex | C1, C2, C10 | C1, C2, C10 |
| aq2 | custom-retrieval-stack | llamaindex | C3, C6, C4, C1, C2 | C3, C4, C6 |
| aq3 | custom-retrieval-stack | llamaindex | C5, C6, C9, C1, C2 | C5, C6 |
| aq4 | custom-retrieval-stack | llamaindex | C10, C2, C4, C3, C8 | C10, C4, C3, C8 |
| aq5 | custom-retrieval-stack | llamaindex | C8, C7 | C8, C7 |
| aq6 | custom-retrieval-stack | llamaindex | C9, C2, C5 | C9, C2, C5 |
| aq7 | custom-retrieval-stack | llamaindex | C5, C6, C10, C8, C2 |  |

## Per-Sample Evidence

### aq1: How do I reset my password?
- **Reference:** Go to account settings, click Forgot password, check your email for a reset link, and follow the link to create a new password.

#### Custom
- **Status:** success
- **Backend:** custom-retrieval-stack
- **Chunks:** C1, C2, C10
- **Answer:** To reset your password:

1. Go to **Account Settings**.  
2. Click the **“Forgot password”** button.  
3. A reset link will be sent to the email address on file.  
4. Open that email and click the link to create a new password and regain access.  

You can also change your password directly from the
- **Error:** None

#### Framework
- **Status:** success
- **Backend:** llamaindex
- **Chunks:** C1, C2, C10
- **Answer:** To reset your password:

1. Go to **Account Settings**.  
2. Click the **“Forgot password”** button.  
3. A reset link will be sent to the email address on file.  
4. Open the email and click the link to create a new password and regain access.  

You can also change your password directly from the 
- **Error:** None

### aq2: How long do billing refunds take to process?
- **Reference:** Billing refunds are processed within 5 business days for eligible charges and are issued to the original payment method.

#### Custom
- **Status:** success
- **Backend:** custom-retrieval-stack
- **Chunks:** C3, C6, C4, C1, C2
- **Answer:** Billing refunds are processed within **5 business days** for eligible charges.
- **Error:** None

#### Framework
- **Status:** success
- **Backend:** llamaindex
- **Chunks:** C3, C4, C6
- **Answer:** Billing refunds are processed within **5 business days** for eligible charges. If you haven’t received the refund after 10 business days, please contact support.
- **Error:** None

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
- **Chunks:** C10, C2, C4, C3, C8
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
- **Chunks:** C5, C6, C10, C8, C2
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
| answer_correctness (custom) | 0.2014 | | 7 | |
| answer_relevancy (custom) | 0.0938 | | 7 | |
| faithfulness (custom) | 0.2484 | | 7 | |
| context_precision (custom) | 0.3571 | | 7 | |
| context_recall (custom) | 0.8571 | | 7 | |
| answer_correctness (framework) | 0.1979 | | 7 | |
| answer_relevancy (framework) | 0.0869 | | 7 | |
| faithfulness (framework) | 0.2199 | | 7 | |
| context_precision (framework) | 0.4524 | | 7 | |
| context_recall (framework) | 0.8571 | | 7 | |

## Genuine Ragas Metrics

**Not available.** Ragas 0.4.x internally references gpt-4o-mini which is not supported by Groq.
Deterministic metrics are used instead.

No NaN scores were treated as valid metrics.