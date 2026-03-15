"""
AI Mock Interview Agent - ADK Implementation
"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm import completion
import json


GROQ_MODEL = "groq/llama-3.3-70b-versatile"


def generate_interview_question(topic: str, difficulty: str, previous_questions: str = "") -> str:
    """Generates a technical interview question based on topic and difficulty using Groq LLM.
    
    Args:
        topic: The interview topic (e.g., 'AI/ML', 'Machine Learning', 'Deep Learning')
        difficulty: The difficulty level ('easy', 'medium', or 'hard')
        previous_questions: Comma-separated list of previously asked questions to avoid repetition
    
    Returns:
        The generated interview question as a string
    """
    previous_context = ""
    if previous_questions:
        previous_context = f"\n\nPreviously asked questions (DO NOT repeat these):\n{previous_questions}"
    
    prompt = f"""You are an expert technical interviewer. Generate ONE high-quality interview question.

Topic: {topic}
Difficulty: {difficulty}
{previous_context}

Requirements:
- Create a thoughtful, realistic interview question
- The question should test both conceptual understanding and practical knowledge
- Make it relevant to real-world {topic} interviews
- Ensure it matches the {difficulty} difficulty level

Return ONLY the question text, nothing else."""

    try:
        response = completion(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        return result
            
    except Exception as e:
        print(f"Error generating question: {e}")
        return f"Explain a key concept in {topic} and provide a practical example."


def evaluate_student_answer(question: str, student_answer: str) -> str:
    """Evaluates a student's answer to an interview question using Groq LLM.
    
    Args:
        question: The interview question that was asked
        student_answer: The student's answer to evaluate
    
    Returns:
        Detailed evaluation with score, strengths, and areas for improvement
    """
    prompt = f"""You are an expert technical interviewer evaluating a candidate's answer.

QUESTION: {question}

CANDIDATE'S ANSWER: {student_answer}

Provide a thorough evaluation including:
1. A score out of 10
2. Specific strengths in the answer
3. Areas for improvement
4. Important points that were missing

Format your response clearly with sections."""

    try:
        response = completion(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        return result
        
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return "Score: 5/10\n\nThe answer shows basic understanding but could benefit from more detail and specific examples."


def generate_feedback(question: str, student_answer: str, evaluation_text: str) -> str:
    """Generates constructive feedback based on the evaluation using Groq LLM.
    
    Args:
        question: The interview question that was asked
        student_answer: The student's answer
        evaluation_text: The evaluation text from the previous step
    
    Returns:
        Constructive feedback with tips and study recommendations
    """
    prompt = f"""You are a supportive technical interview coach providing constructive feedback.

QUESTION: {question}

CANDIDATE'S ANSWER: {student_answer}

EVALUATION: {evaluation_text}

Provide constructive feedback including:
1. Actionable tips to improve their answer
2. Specific suggestions for interview preparation
3. Recommended topics to study next

Be encouraging and specific."""

    try:
        response = completion(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400
        )
        
        result = response.choices[0].message.content.strip()
        return result
        
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return "Keep practicing! Focus on explaining concepts clearly with concrete examples. Review fundamental concepts and practice with more interview questions."


# Create the root agent
# Using OpenAI GPT-4o-mini for main agent (fast and cost-effective)
# Tools use Groq for ultra-fast inference
root_agent = LlmAgent(
    model=LiteLlm(model="gpt-4o-mini"),
    name="mock_interview_agent",
    description="An AI agent that conducts technical mock interviews, evaluates answers, and provides constructive feedback.",
    instruction="""You are an expert AI Mock Interview Agent conducting technical interviews.

Your role is to:
1. Ask relevant technical interview questions based on the specified topic and difficulty
2. Evaluate student answers thoroughly and fairly
3. Provide constructive feedback to help students improve

## Workflow

When conducting an interview, follow this process:

### Starting the Interview
- Use the `generate_interview_question` tool to create a question based on the topic and difficulty
- Present the question clearly to the student
- Store the question for later evaluation

### Evaluating Answers
- When the student provides an answer, use the `evaluate_student_answer` tool
- Analyze the response for correctness, depth, completeness, and use of examples

### Providing Feedback
- Use the `generate_feedback` tool to create constructive feedback
- Highlight strengths and areas for improvement
- Suggest specific topics to study and provide actionable tips

### Continuing the Interview
- After feedback, ask if the student wants another question
- Adjust difficulty based on performance (8-10: hard, 5-7: medium, 0-4: easy)
- Generate new questions avoiding repetition

## Tone and Style
- Be encouraging and supportive
- Provide specific, actionable feedback
- Use clear, professional language
- Act like a helpful mentor, not just an evaluator

Remember: Your goal is to help students improve their interview skills and build confidence!
""",
    tools=[
        generate_interview_question,
        evaluate_student_answer,
        generate_feedback
    ]
)
