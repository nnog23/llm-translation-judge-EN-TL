import json
from llm_api import LLM  # Your wrapper for Qwen2.5-VL-7B-Instruct

class Planner:
    def __init__(self, memory, tools, logger, device="cuda"):
        self.memory = memory
        self.tools = tools
        self.logger = logger
        self.llm = LLM(device=device)  # Initialize your Hugging Face model

    def evaluate(self, sample):
        source = sample['source']
        translation = sample['translation']
        reference = sample.get('reference', "NONE")

        trace = []
        criteria_scores = {}
        explanations = {}

        # Accuracy
        tool_output = self.tools.alignment_check(source, translation)
        score, explanation, reflection = self._evaluate_criterion("accuracy", source, translation, reference, tool_output)
        criteria_scores["accuracy"] = score
        explanations["accuracy"] = explanation
        trace.append(self._trace_entry("accuracy", tool_output, score, explanation, reflection))

        # Fluency
        tool_output = self.tools.grammar_check(translation)
        score, explanation, reflection = self._evaluate_criterion("fluency", source, translation, reference, tool_output)
        criteria_scores["fluency"] = score
        explanations["fluency"] = explanation
        trace.append(self._trace_entry("fluency", tool_output, score, explanation, reflection))

        # Coherence
        score, explanation, reflection = self._evaluate_criterion("coherence", source, translation, reference, {})
        criteria_scores["coherence"] = score
        explanations["coherence"] = explanation
        trace.append(self._trace_entry("coherence", {}, score, explanation, reflection))

        # Cultural Appropriateness
        tool_output = self.tools.cultural_check(source, translation)
        score, explanation, reflection = self._evaluate_criterion("cultural_appropriateness", source, translation, reference, tool_output)
        criteria_scores["cultural_appropriateness"] = score
        explanations["cultural_appropriateness"] = explanation
        trace.append(self._trace_entry("cultural_appropriateness", tool_output, score, explanation, reflection))

        # Guideline Adherence (uses glossary)
        tool_output = self.tools.check_domain_terms(source, translation)
        score, explanation, reflection = self._evaluate_criterion("guideline_adherence", source, translation, reference, tool_output)
        criteria_scores["guideline_adherence"] = score
        explanations["guideline_adherence"] = explanation
        trace.append(self._trace_entry("guideline_adherence", tool_output, score, explanation, reflection))

        # Completeness
        tool_output = self.tools.alignment_check(source, translation)
        score, explanation, reflection = self._evaluate_criterion("completeness", source, translation, reference, tool_output)
        criteria_scores["completeness"] = score
        explanations["completeness"] = explanation
        trace.append(self._trace_entry("completeness", tool_output, score, explanation, reflection))

        # Aggregate final score
        raw_sum = sum(criteria_scores.values())
        overall_score, label = self._normalize_score(raw_sum)

        final_output = {
            "criteria_scores": criteria_scores,
            "raw_sum": raw_sum,
            "overall_score": overall_score,
            "label": label,
            "explanation": explanations
        }

        # Log process
        self.logger.log(sample_id=hash(source), trace=trace, final_output=final_output)
        return final_output

    def _evaluate_criterion(self, criterion, source, translation, reference, tool_output):
        criterion_def = self.memory.criteria.get(criterion, "")

        # Step 1: Initial evaluation with chain-of-thought
        initial_prompt = f"""
You are an impartial expert bilingual judge evaluating English to Filipino translations.
You are tasked with evaluating the criterion '{criterion}' for a Filipino translation.

Criterion Definition:
{criterion_def}

Tool Output (evidence to consider):
{json.dumps(tool_output, ensure_ascii=False)}

Source Text (English):
{source}

Translation (Filipino):
{translation}

Reference Translation (if provided):
{reference}

Instructions:
- Think step by step about this evaluation.
- First, analyze what you observe in the translation relative to the criterion.
- Consider the tool output evidence carefully.
- Then assign a score of 0 or 1 based solely on the criterion definition and evidence.
- Provide a concise explanation (1–2 sentences) supporting your score.
- Remain objective and impartial.

Respond strictly in this JSON format:
{{
    "reasoning": "<your step-by-step thinking process>",
    "score": 0 or 1,
    "explanation": "<brief, clear explanation>"
}}
"""

        # Call the LLM for initial evaluation
        initial_response = None
        for attempt in range(3):
            try:
                raw_response = self.llm.call(initial_prompt, max_new_tokens=400, temperature=0.0)
                initial_response = json.loads(raw_response.strip())
                if all(key in initial_response for key in ["reasoning", "score", "explanation"]):
                    break
            except json.JSONDecodeError:
                if attempt == 2:
                    return 0, "Invalid JSON after 3 attempts", "Failed to parse initial evaluation"
                continue

        if not initial_response:
            return 0, "Unable to get valid response from LLM", "No initial response received"

        # Step 2: Reflection prompt for internal thoughts
        reflection_prompt = f"""
You are the same bilingual judge who just evaluated the criterion '{criterion}'.

Your previous evaluation:
- Reasoning: {initial_response['reasoning']}
- Score: {initial_response['score']}
- Explanation: {initial_response['explanation']}

Now, reflect on your evaluation process and thoughts:
- What aspects of this evaluation were most challenging or uncertain?
- Are there any potential biases or assumptions you made?
- What additional context or information would have been helpful?
- How confident are you in this assessment?
- What are your internal thoughts about the quality of this translation for this specific criterion?

Provide your honest internal reflection as a judge. This is for your own processing and improvement.

Respond in this JSON format:
{{
    "reflection": "<your internal thoughts and self-assessment of the evaluation process>"
}}
"""

        # Get reflection
        reflection = "No reflection generated"
        for attempt in range(2):  # Fewer attempts for reflection to maintain speed
            try:
                raw_reflection = self.llm.call(reflection_prompt, max_new_tokens=300, temperature=0.1)
                reflection_response = json.loads(raw_reflection.strip())
                if "reflection" in reflection_response:
                    reflection = reflection_response["reflection"]
                    break
            except json.JSONDecodeError:
                if attempt == 1:
                    reflection = f"Reflection parsing failed. Initial reasoning was: {initial_response.get('reasoning', 'N/A')}"
                continue

        return int(initial_response["score"]), initial_response["explanation"], reflection

    def _trace_entry(self, criterion, tool_output, score, explanation, reflection):
        return {
            "criterion": criterion,
            "tool_output": tool_output,
            "score": score,
            "explanation": explanation,
            "reflection": reflection
        }

    def _normalize_score(self, raw_sum):
        if raw_sum >= 5:
            return 5, "excellent"
        elif raw_sum >= 3:
            return raw_sum, "good"
        else:
            return raw_sum, "poor"