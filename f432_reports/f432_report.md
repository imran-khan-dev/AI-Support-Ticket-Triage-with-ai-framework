# Answer Quality Comparison Report

- **Experiment ID:** e99bbe8c-5d78-42ab-b31b-caa0f001971f
- **Dataset:** support-kb-answer-quality-eval (16dafdb5-f374-42a9-880e-1964cf35ec1c)
- **Status:** completed
- **Completed at:** 2026-11-23T12:22:36.972783+00:00
- **Top-k:** 5
- **Tenant:** eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee
- **Ragas available:** False

## Aggregate Results

| Metric | Custom Mean | Custom Count | Framework Mean | Framework Count |
|--------|-------------|--------------|----------------|-----------------|
| answer_correctness | 0.5984 | 7 | 0.5219 | 7 |
| answer_relevancy | 0.2961 | 7 | 0.2394 | 7 |
| faithfulness | 0.4884 | 7 | 0.5335 | 6 |
| context_precision | 0.3571 | 7 | 0.4524 | 7 |
| context_recall | 0.8571 | 7 | 0.8571 | 7 |

- **Custom completed:** 7
- **Custom failed:** 0
- **Framework completed:** 7
- **Framework failed:** 0

## Per-Sample Results

### aq1: How do I reset my password?
- **Category:** direct_factual
- **Sufficient context:** True
- **Reference:** Go to account settings, click Forgot password, check your email for a reset link, and follow the link to create a new password.

#### Custom
- **Status:** success
- **Answer:** To reset your password:

1. Go to **Account Settings**.  
2. Click the **“Forgot password”** button.  
3. A reset link will be sent to the email address on file.  
4. Open that email and click the link to create a new password and regain access.  

You can also change your password directly from the Account Settings page if you’re already logged in.
- **Context chunks:** ['C1', 'C2', 'C10']
- **Metrics:**
  - answer_correctness: 0.7029
  - answer_relevancy: 0.2242
  - faithfulness: 0.7391
  - context_precision: 0.6667
  - context_recall: 1.0000

#### Framework
- **Status:** success
- **Answer:** To reset your password:

1. Go to **Account Settings**.  
2. Click the **“Forgot password”** button.  
3. A reset link will be sent to the email address on file.  
4. Open that email and click the link to create a new password and regain access.  

You can also change your password directly from the Account Settings page if you’re already logged in.
- **Context chunks:** ['C1', 'C2', 'C10']
- **Metrics:**
  - answer_correctness: 0.7029
  - answer_relevancy: 0.2242
  - faithfulness: 0.7391
  - context_precision: 0.6667
  - context_recall: 1.0000

### aq2: How long do billing refunds take to process?
- **Category:** direct_factual
- **Sufficient context:** True
- **Reference:** Billing refunds are processed within 5 business days for eligible charges and are issued to the original payment method.

#### Custom
- **Status:** success
- **Answer:** Billing refunds are processed within **5 business days** for eligible charges.
- **Context chunks:** ['C3', 'C6', 'C4', 'C1', 'C2']
- **Metrics:**
  - answer_correctness: 0.7071
  - answer_relevancy: 0.4326
  - faithfulness: 1.0000
  - context_precision: 0.4000
  - context_recall: 1.0000

#### Framework
- **Status:** success
- **Answer:** Billing refunds are processed within **5 business days** for eligible charges.
- **Context chunks:** ['C3', 'C4', 'C6']
- **Metrics:**
  - answer_correctness: 0.7071
  - answer_relevancy: 0.4326
  - faithfulness: 1.0000
  - context_precision: 0.6667
  - context_recall: 1.0000

### aq3: How long does shipping take and do I get tracking information?
- **Category:** direct_factual
- **Sufficient context:** True
- **Reference:** Orders ship within 24 hours and tracking information is provided via email.

#### Custom
- **Status:** success
- **Answer:** Shipping takes up to 24 hours, and you’ll receive tracking information via email. (International orders may take a bit longer.)
- **Context chunks:** ['C5', 'C6', 'C9', 'C1', 'C2']
- **Metrics:**
  - answer_correctness: 0.7426
  - answer_relevancy: 0.3352
  - faithfulness: 0.6923
  - context_precision: 0.2000
  - context_recall: 1.0000

#### Framework
- **Status:** success
- **Answer:** Shipping takes up to 24 hours to ship out, and you’ll receive tracking information via email once the order is dispatched. Shipping is free for orders over $50, and international shipments may take additional time.
- **Context chunks:** ['C5', 'C6']
- **Metrics:**
  - answer_correctness: 0.6696
  - answer_relevancy: 0.2353
  - faithfulness: 0.6667
  - context_precision: 0.5000
  - context_recall: 1.0000

### aq4: Can I get a refund for a purchase from 6 months ago?
- **Category:** insufficient_context
- **Sufficient context:** False
- **Reference:** The knowledge base does not state a time limit for requesting refunds. Billing refunds are processed within 5 business days for eligible charges. Contact support if your refund has not arrived after 10 business days.

#### Custom
- **Status:** success
- **Answer:** I’m sorry, but the provided information doesn’t specify whether refunds can be issued for purchases made as far back as six months ago. You may want to contact support directly to confirm the refund policy for older purchases.
- **Context chunks:** ['C10', 'C2', 'C4', 'C3', 'C8']
- **Metrics:**
  - answer_correctness: 0.0543
  - answer_relevancy: 0.1293
  - faithfulness: 0.2174
  - context_precision: 0.4000
  - context_recall: 1.0000

#### Framework
- **Status:** success
- **Answer:** I’m sorry, but the provided information doesn’t specify whether refunds can be issued for purchases made as far back as six months ago. You may want to reach out to support for clarification.
- **Context chunks:** ['C10', 'C4', 'C3', 'C8']
- **Metrics:**
  - answer_correctness: 0.0344
  - answer_relevancy: 0.1002
  - faithfulness: 0.1500
  - context_precision: 0.5000
  - context_recall: 1.0000

### aq5: How do I check my API rate limit usage?
- **Category:** direct_factual
- **Sufficient context:** True
- **Reference:** You can view your rate limit usage in the developer dashboard.

#### Custom
- **Status:** success
- **Answer:** You can view your API rate‑limit usage directly in the **Developer Dashboard**. Log in to your account, navigate to the dashboard, and look for the rate‑limit or usage section to see how many requests you’ve made and how many remain.
- **Context chunks:** ['C8', 'C7']
- **Metrics:**
  - answer_correctness: 0.7034
  - answer_relevancy: 0.2791
  - faithfulness: 0.3889
  - context_precision: 0.5000
  - context_recall: 1.0000

#### Framework
- **Status:** success
- **Answer:** You can view your API rate‑limit usage directly in the **Developer Dashboard**. Log in to your account, navigate to the dashboard, and look for the rate‑limit or usage section to see how many requests you’ve made and how many remain.
- **Context chunks:** ['C8', 'C7']
- **Metrics:**
  - answer_correctness: 0.7034
  - answer_relevancy: 0.2791
  - faithfulness: 0.3889
  - context_precision: 0.5000
  - context_recall: 1.0000

### aq6: How do I update my email notification preferences?
- **Category:** direct_factual
- **Sufficient context:** True
- **Reference:** Account settings let you update your profile, email address, and notification preferences.

#### Custom
- **Status:** success
- **Answer:** To update your email notification preferences, follow these steps:

1. **Log in** to your account on the website or app.  
2. Navigate to **Account Settings** (often found in the profile menu or sidebar).  
3. Within Account Settings, locate the **Notification Preferences** section.  
4. Adjust the options for email notifications to your liking—enable or disable specific alerts, change frequency, etc.  
5. **Save** your changes.

After saving, the new preferences will take effect immediately.
- **Context chunks:** ['C9', 'C2', 'C5']
- **Metrics:**
  - answer_correctness: 0.4517
  - answer_relevancy: 0.1117
  - faithfulness: 0.2564
  - context_precision: 0.3333
  - context_recall: 1.0000
- **Unsupported claims:** ['**Save** your changes']

#### Framework
- **Status:** success
- **Answer:** To update your email notification preferences, follow these steps:

1. **Log in** to your account on the website or app.  
2. Navigate to **Account Settings** (often found in the profile menu or sidebar).  
3. Within Account Settings, locate the **Notification Preferences** section.  
4. Adjust the options for email notifications to your liking—enable or disable specific alerts, change frequency, etc.  
5. **Save** your changes.

After saving, the new preferences will take effect immediately.
- **Context chunks:** ['C9', 'C2', 'C5']
- **Metrics:**
  - answer_correctness: 0.4517
  - answer_relevancy: 0.1117
  - faithfulness: 0.2564
  - context_precision: 0.3333
  - context_recall: 1.0000
- **Unsupported claims:** ['**Save** your changes']

### aq7: What is the company's stock price?
- **Category:** no_context
- **Sufficient context:** False
- **Reference:** I don't have information about the company's stock price.

#### Custom
- **Status:** success
- **Answer:** I’m sorry, but I don’t have any information about the company’s stock price.
- **Context chunks:** ['C5', 'C6', 'C10', 'C8', 'C2']
- **Metrics:**
  - answer_correctness: 0.8271
  - answer_relevancy: 0.5606
  - faithfulness: 0.1250
  - context_precision: 0.0000
  - context_recall: 0.0000

#### Framework
- **Status:** success
- **Answer:** I’m sorry, but I don’t have access to real‑time stock market data. I can’t provide the current price of the company’s shares.
- **Context chunks:** []
- **Metrics:**
  - answer_correctness: 0.3846
  - answer_relevancy: 0.2923
  - faithfulness: N/A
  - context_precision: 0.0000
  - context_recall: 0.0000
- **Unsupported claims:** ['I’m sorry, but I don’t have access to real‑time stock market data', 'I can’t provide the current price of the company’s shares']

## Limitations

- This is a small, deterministic evaluation dataset.
- No statistical significance claims are made from these results.
- Deterministic faithfulness uses keyword overlap, not LLM judgment. Ragas faithfulness (when available) uses LLM-based evaluation.
- Mock mode (no LLM client) returns deterministic mock answers.