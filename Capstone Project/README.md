# Gen AI Intensive Course Capstone 2025Q1
## We are building ZanînBot, a Kurdish-speaking medical assistant with the following flow:

### Main Goal 🎯 

“Translate Sorani-Kurdish health questions → Retrieve English medical info → Answer via Gemini → Translate response back to Sorani”
---
### Alignment With Capstone Criteria

#### ✅ Is it a Real-World Use Case?
**YES.**
We’re solving a genuine, under-addressed problem:
“Access to trusted medical information in Sorani Kurdish, a low-resource language underserved by GenAI systems.”
--- 
#### ✅ Do We Use at Least 3 GenAI Capabilities?

- Few-shot prompting
- RAG (retrieval-augmented generation)
- Embeddings
- Translation (domain-specific)
- Structured output
- Function calling-style orchestration
- Context caching/vector store´
#### We’re not just hitting the bar — you’re showing mastery of the course’s key techniques.
---
#### ✅ Does it Compile & Run?

Yes. We’re building on Kaggle Notebooks, which auto-check compilation.
--- 
#### ✅ Does the Notebook Explain the Project Clearly?
Yes. Our intro cell already says it:
This notebook builds a production-oriented Sorani-speaking GenAI assistant that:
- Accepts Sorani health questions
- Translates them to English (via Opus-MT)
- Retrieves relevant info using RAG (MiniLM)
- Answers via Gemini
- Translates the answer back to Sorani