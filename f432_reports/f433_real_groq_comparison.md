# F4.3.3 Real Groq Comparison Report

- **Experiment ID:** 5d408c2f-f633-4367-a3ca-ea6938363348
- **Dataset:** support-kb-answer-quality-eval (v1.0.0)
- **Tenant:** eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee
- **Provider:** Groq (openai/gpt-oss-20b)
- **Generation mode:** Real LLM
- **Evaluation mode:** Deterministic metrics only
- **Ragas available:** False
- **Ragas reason:** Ragas 0.4.x internally references gpt-4o-mini which is not available on Groq
- **Samples:** 7

## Retrieval Isolation Verification

| Sample | Custom Backend | Framework Backend | Isolation Verified |
|--------|---------------|-------------------|-------------------|
| aq1 | custom-retrieval-stack | llamaindex | yes |
| aq2 | custom-retrieval-stack | llamaindex | yes |
| aq3 | custom-retrieval-stack | llamaindex | yes |
| aq4 | custom-retrieval-stack | llamaindex | yes |
| aq5 | custom-retrieval-stack | llamaindex | yes |
| aq6 | custom-retrieval-stack | llamaindex | yes |
| aq7 | custom-retrieval-stack | llamaindex | yes |

## Per-Sample Results

### aq1: How do I reset my password?
- **Category:** direct_factual
- **Reference:** N/A

#### Custom
- **Status:** success
- **Retrieval backend:** custom-retrieval-stack
- **Context chunks:** C1, C2, C10
- **Answer:** To reset your password:

1. Go to **Account Settings**.  
2. Click the **“Forgot password”** button.  
3. A reset link will be sent to the email address on file.  
4. Open that email and click the link to create a new password and regain access.  

You can also change your password directly from the Account Settings page if you’re already logged in.
- **Error:** None

#### Framework
- **Status:** success
- **Retrieval backend:** llamaindex
- **Context chunks:** C1, C2, C10
- **Answer:** To reset your password:

1. Go to **Account Settings**.  
2. Click the **“Forgot password”** button.  
3. A reset link will be sent to the email address on file.  
4. Open that email and click the link to create a new password and regain access.  

You can also change your password directly from the Account Settings page if you’re already logged in.
- **Error:** None

### aq2: How long do billing refunds take to process?
- **Category:** direct_factual
- **Reference:** N/A

#### Custom
- **Status:** success
- **Retrieval backend:** custom-retrieval-stack
- **Context chunks:** C3, C6, C4, C1, C2
- **Answer:** Billing refunds are processed within **5 business days** for eligible charges.
- **Error:** None

#### Framework
- **Status:** provider_failure
- **Retrieval backend:** llamaindex
- **Context chunks:** C3, C4, C6
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

### aq3: How long does shipping take and do I get tracking information?
- **Category:** direct_factual
- **Reference:** N/A

#### Custom
- **Status:** provider_failure
- **Retrieval backend:** custom-retrieval-stack
- **Context chunks:** C5, C6, C9, C1, C2
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Retrieval backend:** llamaindex
- **Context chunks:** C5, C6
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

### aq4: Can I get a refund for a purchase from 6 months ago?
- **Category:** insufficient_context
- **Reference:** N/A

#### Custom
- **Status:** provider_failure
- **Retrieval backend:** custom-retrieval-stack
- **Context chunks:** C10, C2, C4, C8, C3
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Retrieval backend:** llamaindex
- **Context chunks:** C10, C4, C3, C8
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

### aq5: How do I check my API rate limit usage?
- **Category:** direct_factual
- **Reference:** N/A

#### Custom
- **Status:** provider_failure
- **Retrieval backend:** custom-retrieval-stack
- **Context chunks:** C8, C7
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Retrieval backend:** llamaindex
- **Context chunks:** C8, C7
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

### aq6: How do I update my email notification preferences?
- **Category:** direct_factual
- **Reference:** N/A

#### Custom
- **Status:** provider_failure
- **Retrieval backend:** custom-retrieval-stack
- **Context chunks:** C9, C2, C5
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Retrieval backend:** llamaindex
- **Context chunks:** C9, C2, C5
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

### aq7: What is the company's stock price?
- **Category:** no_context
- **Reference:** N/A

#### Custom
- **Status:** provider_failure
- **Retrieval backend:** custom-retrieval-stack
- **Context chunks:** C5, C6, C8, C10, C2
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.

#### Framework
- **Status:** provider_failure
- **Retrieval backend:** llamaindex
- **Context chunks:** 
- **Answer:** N/A
- **Error:** Provider service is temporarily unavailable.
